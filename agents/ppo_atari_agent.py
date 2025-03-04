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

from models.ppo_atari import ActorCritic
from data.transition import PPOTransition 
from utils.tech_utils import *
from utils.common_utils import log_callback, print_callback, save_state, get_time, get_time_diff

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
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

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

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = ActorCritic(
            env.single_action_space.n, activation=config["ACTIVATION"]
        )

        def create_agent(rng, init_network):
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            network_variables = init_network.init(rng, init_x)

            tabulate_fn = nn.tabulate(network, jax.random.PRNGKey(config["SEED"]))
            # print(tabulate_fn(init_x))

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables,
                batch_stats=None,
                tx=tx,
            )

            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng, network)

        global_st = get_time()

        # TRAINING LOOP
        def _update_step(runner_state, unused): 

            train_state, expl_state, global_metrics, rng = runner_state

            metrics = {"general": {}, "losses": {}, "time": {}, "misc": {}}

            metrics["misc"].update({"prev_steps": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"]})





            # EVALUATE AGENT
            st = get_time()

            rng, _rng = jax.random.split(rng)
            eval_metrics = jax.lax.cond(jnp.logical_or(train_state.n_updates % int(config["NUM_UPDATES"]*config["EVAL_INTERVAL"]) == 0, 
                                                       train_state.n_updates == config["NUM_UPDATES"]), 
                                        lambda _: eval_agent(train_state, _rng), lambda _: global_metrics["eval"], operand=None)
            
            global_metrics["eval"] = eval_metrics

            metrics["time"].update({"eval_time": get_time_diff(get_time(), st)})





            # SAMPLE PHASE
            st = get_time()

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                new_action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(new_action)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = PPOTransition(
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    obs=last_obs,
                    value=value,
                    log_prob=log_prob,
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
            last_obs, _ = expl_state

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )
                last_obs = last_obs[: -config["TEST_ENVS"]]

            transitions = PPOTransition(
                    obs=transitions.obs,
                    action=transitions.action,
                    reward=transitions.reward,
                    done=transitions.done,
                    value=transitions.value,
                    log_prob=transitions.log_prob,
                )

            metrics["time"].update({"env_time": get_time_diff(get_time(), st)})






            # NETWORKS UPDATE
            st = get_time()

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(transitions, last_val)


            def preprocess_transition(x, rng):
                x = x.reshape(
                    -1, *x.shape[2:]
                )  # num_steps*num_envs (batch_size), ...
                x = jax.random.permutation(rng, x)  # shuffle the transitions
                x = x.reshape(
                    config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                )  # num_mini_updates, batch_size/num_mini_updates, ...
                return x
            

            def _learn_epoch(carry, _):
                train_state, traj_batch, advantages, targets, rng = carry

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                

                rng, _rng = jax.random.split(rng)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, (traj_batch, advantages, targets)
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)

                return update_state, total_loss



            # Create rng key only once for minibatches, targets and int_targets to maintain consistency
            rng, _rng = jax.random.split(rng)
            minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
            targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), targets
                )
            advantages = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), advantages
                )
            
            update_state = (train_state, minibatches, advantages, targets, rng)
            

            rng, _rng = jax.random.split(rng)
            (train_state, _, _, _, rng), loss_info = jax.lax.scan(
                _learn_epoch, update_state, None, config["NUM_EPOCHS"]
            )


            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics["time"].update({"train_time": get_time_diff(get_time(), st)})






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
            total_loss, (value_loss, loss_actor, entropy) = loss_info
            metrics["losses"].update({
                "value_loss": value_loss.mean(),
                "actor_loss": loss_actor.mean(),
                "entropy": entropy.mean(),
                "total_loss": total_loss.mean(),
                "reward": transitions.reward.mean(),
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
                pi, value = network.apply(train_state.params, last_obs)
                new_action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(new_action)

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
                "eval_returned_episode_returns": eval_state.returned_episode_returns.mean(),
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
        global_metrics.update({"eval": {
            "eval_returned_episode_returns": 0.0,
            "eval_normalized_returned_episode_returns": 0.0,
            "eval_returned_episode_lengths": 0.0,
        }})
        

        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (train_state, expl_state, global_metrics, _rng)


        if config.get("SAVE_STATE", True):

            ckpt = {"runner_state": train_state, "metrics": global_metrics}
            jax.debug.callback(save_state, ckpt, 0)



        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

