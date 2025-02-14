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

from models.pqn_rnn_craftax import RNNQNetwork
from data.transition import RNNTransition 
from utils.common_utils import log_callback, save_state, print_dashboard, get_time, get_time_diff

from envs.craftax import make_env

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

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"


    env , test_env, env_params = make_env(config["ENV_NAME"], config["NUM_ENVS"], config["TEST_NUM_ENVS"],
                                          config["OPTIMISTIC_RESET_RATIO"], config["USE_OPTIMISTIC_RESETS"])


    def train(rng):
        
        original_rng = rng[0]
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
        network = RNNQNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, *env.observation_space(env_params).shape)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )  # (obs, dones, last_actions)
            init_hs = network.initialize_carry(1)  # (batch_size, hidden_dim)
            network_variables = network.init(rng, init_hs, *init_x, train=False)

            tabulate_fn = nn.tabulate(network, jax.random.PRNGKey(0))
            print(tabulate_fn(init_hs, *init_x, train=False))

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

            train_state, memory_transitions, expl_state, global_metrics, rng = (
                runner_state
            )

            metrics = {"general": {}, "losses": {}, "time": {}, "misc": {}}

            metrics["misc"].update({"prev_steps": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"]})




            # SAMPLE PHASE
            st = get_time()

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                transition = RNNTransition(
                    last_hs=hs,
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                )
                return (new_hs, new_obs, new_done, new_action, new_env_state, rng), (
                    transition,
                    info,
                )

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            metrics["time"].update({"env_time": get_time_diff(get_time(), st)})




            # NETWORKS UPDATE
            st = get_time()

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # insert the transitions into the memory
            memory_transitions = jax.tree_map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):

                    # minibatch shape: num_steps, batch_size, ...
                    # with batch_size = num_envs/num_minibatches

                    train_state, rng = carry
                    hs = jax.tree_map(lambda x: x[0], minibatch.last_hs)  # hs of oldest step (batch_size, hidden_size)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )

                    def _compute_targets(last_q, q_vals, reward, done):
                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                target_bootstrap
                                + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (1 - done) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            return (lambda_returns, next_q), lambda_returns

                        lambda_returns = (
                            reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                        )
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                            reverse=True,
                        )
                        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                        return targets

                    def _loss_fn(params):
                        (_, q_vals), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                        )  # (num_steps, batch_size, num_actions)

                        # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        target = _compute_targets(
                            last_q,  # q_vals at t=NUM_STEPS-1
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(
                            -1
                        )  # (num_steps-1*batch_size,)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (num_steps, num_agents, batch_size,)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(
                            -1
                        )  # (num_steps-1*batch_size,)

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
                    # x: (num_steps, num_envs, ...)
                    x = jax.random.permutation(
                        rng, x, axis=1
                    )  # shuffle the transitions
                    x = x.reshape(
                        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                    )  # num_steps, minibatches, batch_size/num_minbatches,
                    x = jnp.swapaxes(
                        x, 0, 1
                    )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    memory_transitions,
                )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            done_infos = jax.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )

            metrics["time"].update({"train_time": get_time_diff(get_time(), st)})




            # EVALUATE AGENT
            st = get_time()
            
            # Cannot evaluate the entire episode of Craftax, but we can evaluate in small num of steps continually

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: global_metrics["eval"],
                    operand=None,
                )
                # metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
                # global_metrics["eval"] = {f"test_{k}": v for k, v in test_metrics.items()}
                global_metrics["eval"] = test_metrics

            # remove achievement metrics if not logging them
            # if not config.get("LOG_ACHIEVEMENTS", False):
            #     metrics["general"].update({
            #         k: v for k, v in metrics.items() if "achievement" not in k.lower()
            #     })

            metrics["time"].update({"eval_time": get_time_diff(get_time(), st)})





            # REPORT METRICS
            metrics["general"].update({
                "iterations": train_state.n_updates,
                "env_steps": train_state.timesteps,
                "env_frames": train_state.timesteps,
                "grad_steps": train_state.grad_steps,
            })
            metrics["losses"].update({
                "q_value_loss": loss.mean(),
                "q_values": qvals.mean(),
            })

            # metrics["general"].update({k: v.mean() for k, v in infos.items()})
            metrics["general"].update(done_infos)

            metrics["time"].update({"uptime": get_time_diff(get_time(), global_st)})
            metrics["misc"].update({"start_time": global_st})

            # report statistics
            log_callback(metrics, global_metrics, original_rng, config)

            global_metrics["env_time"] += metrics["time"]["env_time"]
            global_metrics["eval_time"] += metrics["time"]["eval_time"]
            global_metrics["train_time"] += metrics["time"]["train_time"]
            global_metrics["uptime"] = metrics["time"]["uptime"]

            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                global_metrics,
                rng,
            )

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

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(step_state, _):
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                return step_state, info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
            init_hs = network.initialize_carry(
                config["TEST_NUM_ENVS"]
            )  # (n_envs, hs_size)
            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            step_state, infos = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )

            done_infos = {f"test_{k}": v for k, v in done_infos.items()}

            return done_infos
        

        # train
        global_metrics = dict(
            env_time = 0.0,
            eval_time = 0.0,
            train_time = 0.0,
            uptime = 0.0
        )

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)
        global_metrics.update({"eval": test_metrics})


        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
        init_hs = network.initialize_carry(config["NUM_ENVS"])
        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        # step randomly to have the initial memory window
        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
            new_hs, q_vals = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
            )  # (num_envs, hidden_size), (1, num_envs, num_actions)
            q_vals = q_vals.squeeze(
                axis=0
            )  # (num_envs, num_actions) remove the time dim
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )
            transition = RNNTransition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_vals=q_vals,
            )
            return (
                new_hs,
                new_obs,
                new_done,
                new_action,
                new_env_state,
                rng,
            ), transition

        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        expl_state = tuple(expl_state)


        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, global_metrics, _rng)

        if config.get("SAVE_STATE", True):
                
            ckpt = {"runner_state": train_state, "metrics": global_metrics}
            jax.debug.callback(save_state, ckpt, 0)


        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

