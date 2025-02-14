import os
import time

import jax.experimental
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import flax.linen as nn
import optax

from functools import partial

from typing import Any

from models.pqn_cnn import QNetwork
from data.transition import Transition 
from utils.common_utils import log_callback, save_state, print_dashboard, get_time, get_time_diff

from envs.envpool import make_env

from exploration.eps_greedy import eps_greedy_exploration


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    print(config["NUM_UPDATES"])

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    total_eval_envs = config["EVAL_NUM_ENVS"]

    env = make_env(total_envs, config["ENV_NAME"], config["SEED"], config["ENV_KWARGS"])
    
    eval_env = make_env(total_eval_envs, config["ENV_NAME"], config["SEED"], config["ENV_KWARGS"])

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng):
        
        original_seed = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.single_action_space.n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            network_variables = network.init(rng, init_x, train=False)

            tabulate_fn = nn.tabulate(network, jax.random.PRNGKey(0))
            print(tabulate_fn(init_x, train=False))

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )

            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        global_st = get_time()

        # TRAINING LOOP
        def _update_step(runner_state, unused): 

            train_state, expl_state, global_metrics, rng = runner_state

            metrics = {"general": {}, "losses": {}, "time": {}, "misc": {}}

            metrics["misc"].update({"prev_steps": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"]})


            # SAMPLE PHASE
            st = get_time()

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )

            metrics["time"].update({"env_time": get_time_diff(get_time(), st)})




            # NETWORKS UPDATE
            st = get_time()

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                last_q = jnp.max(q_vals[-1], axis=-1)
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            lambda_targets = _compute_targets(
                last_q, transitions.q_val, transitions.reward, transitions.done
            )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics["time"].update({"train_time": get_time_diff(get_time(), st)})




            # EVALUATE AGENT
            st = get_time()
            
            eval_metrics = jax.lax.cond(jnp.logical_or(train_state.n_updates % int(config["NUM_UPDATES"]*config["EVAL_INTERVAL"]) == 0, 
                                                       train_state.n_updates == config["NUM_UPDATES"]), 
                                        lambda _: eval_agent(train_state, _rng), lambda _: global_metrics["eval"], operand=None)
            
            global_metrics["eval"] = eval_metrics

            metrics["time"].update({"eval_time": get_time_diff(get_time(), st)})





            # REPORT METRICS
            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree_map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
                infos = jax.tree_map(lambda x: x[:, : -config["TEST_ENVS"]], infos)
                infos.update({"test_" + k: v for k, v in test_infos.items()})

            metrics["general"].update({
                "iterations": train_state.n_updates,
                "env_steps": train_state.timesteps,
                "env_frames": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"],
                "grad_steps": train_state.grad_steps,
            })
            metrics["losses"].update({
                "q_value_loss": loss.mean(),
                "q_values": qvals.mean(),
            })

            metrics["general"].update({k: v.mean() for k, v in infos.items()})

            metrics["time"].update({"uptime": get_time_diff(get_time(), global_st)})
            metrics["misc"].update({"start_time": global_st})

            if config.get("TEST_DURING_TRAINING", False):
                metrics["general"].update({f"test_{k}": v.mean() for k, v in test_infos.items()})

            # report statistics
            log_callback(metrics, global_metrics, original_seed, config)

            global_metrics["env_time"] += metrics["time"]["env_time"]
            global_metrics["eval_time"] += metrics["time"]["eval_time"]
            global_metrics["train_time"] += metrics["time"]["train_time"]
            global_metrics["uptime"] = metrics["time"]["uptime"]

            runner_state = (train_state, tuple(expl_state), global_metrics, rng)

            if config.get("SAVE_STATE", True):
                
                ckpt = {"runner_state": train_state, "metrics": metrics}
                def save_callback(ckpt, iteration):
                    jax.debug.callback(save_state, ckpt, iteration)
                    return
                def save_nothing(ckpt, iteration):
                    return

                jax.lax.cond(jnp.logical_or(train_state.n_updates % int(config["NUM_UPDATES"]*config["SAVE_INTERVAL"]) == 0, 
                                            train_state.n_updates == config["NUM_UPDATES"]), 
                             save_callback, save_nothing, ckpt, metrics["general"]["iterations"])

            return runner_state, metrics

        def eval_agent(train_state, rng):

            # EVAL PHASE
            def _step_env(carry):
                last_obs, env_state, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                _rngs = jax.random.split(rng, total_eval_envs)
                eps = jnp.full(config["EVAL_NUM_ENVS"], config["EPS_EVAL"]) 
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, done, info = eval_env.step(
                    env_state, new_action
                )

                return (new_obs, new_env_state, rng)

            rng, _rng = jax.random.split(rng)
            eval_obs, eval_state = eval_env.reset()
            (eval_obs, eval_state, _rng) = jax.lax.while_loop(
                lambda x: jnp.any(x[1].returned_episode_lengths == 0),  # Only when agent completes an episode, the eps length will be greater than 0
                _step_env,
                ((eval_obs, eval_state, _rng)),
            )
            normalize_score = lambda x: (x - eval_env.env_random_score) / (
                    eval_env.env_human_score - eval_env.env_random_score
                )
            eval_metrics = {
                "eval_returned_episode_returns": eval_state.returned_episode_lengths.mean(),
                "eval_normalized_returned_episode_returns": normalize_score(eval_state.returned_episode_returns.mean()),
                "eval_returned_episode_lengths": eval_state.returned_episode_lengths.mean()
            }
            return eval_metrics
        

        # train
        global_metrics = dict(
            env_time = 0.0,
            eval_time = 0.0,
            train_time = 0.0,
            uptime = 0.0
        )

        rng, _rng = jax.random.split(rng)
        eval_metrics = eval_agent(train_state, _rng)
        global_metrics.update({"eval": eval_metrics})
        

        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (train_state, expl_state, global_metrics, _rng)


        if config.get("SAVE_STATE", True):

            ckpt = {"runner_state": train_state, "metrics": metrics}
            jax.debug.callback(save_state, ckpt, 0)



        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

