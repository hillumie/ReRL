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

from models.ppo_rnd_cnn_2v import ActorCritic, FeatureRNDCNN, TargetFeatureRNDCNN
from data.transition import PPORND2vTransition 
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

    config["NUM_RANDOM"] = int(
        config["NUM_UPDATES"] * config["RANDOM_WALK_PROPORTION"]
    )

    config["NUM_UPDATES"] = config["NUM_UPDATES"] - config["NUM_RANDOM"]

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

        rnd_network = FeatureRNDCNN()

        target_rnd_network = TargetFeatureRNDCNN()

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
        
        def create_rnd(rng, init_network):
            init_x = jnp.zeros((1, 1, *env.single_observation_space.shape[1:]))
            network_variables = init_network.init(rng, init_x)

            tabulate_fn = nn.tabulate(init_network, jax.random.PRNGKey(config["SEED"]))
            print(tabulate_fn(init_x))

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=init_network.apply,
                params=network_variables["params"],
                batch_stats=None,
                tx=tx,
            )

            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng, network)

        rng, _rng = jax.random.split(rng)
        rnd_train_state = create_rnd(rng, rnd_network)

        rng, _rng = jax.random.split(rng)
        target_rnd_train_state = create_rnd(rng, target_rnd_network)

        obs_rms = RunningMeanStd(
            mean = jnp.zeros((1, 1, 84, 84)),
            var = jnp.ones((1, 1, 84, 84)),
            count = 1e-4,
        )

        reward_rms = RunningMeanStd(
            mean = jnp.zeros(1),
            var = jnp.ones(1),
            count = 1e-4,
        )

        rewems = jnp.zeros(config["NUM_ENVS"])

        global_st = get_time()

        # TRAINING LOOP
        def _update_step(runner_state, unused): 

            full_train_state, expl_state, global_metrics, rng = runner_state

            train_state, rnd_train_state, obs_rms, reward_rms, rewems = full_train_state


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
                pi, ext_value, int_value = network.apply(train_state.params, last_obs)
                new_action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(new_action)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )
                rnd_new_obs = ((
                    (new_obs[:, -1, :, :].reshape((-1, 1,) + env.observation_space.shape[1:]) - obs_rms.mean) / jnp.sqrt(obs_rms.var)
                                ).clip(-5, 5)).astype(jnp.float32)

                target_next_feature = target_rnd_network.apply(
                    {
                        "params": target_rnd_train_state.params,
                    },
                    rnd_new_obs,
                    train=False
                )
                predict_next_feature = rnd_network.apply(
                    {
                        "params": rnd_train_state.params,
                    },
                    rnd_new_obs,
                    train=False
                )
                curiosity_reward = ((jnp.square(target_next_feature - predict_next_feature)).sum(axis=1) / 2)

                transition = PPORND2vTransition(
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    obs=last_obs,
                    ext_value=ext_value,
                    int_value=int_value,
                    log_prob=log_prob,
                    curiosity_reward=config.get("REW_SCALE", 1) * curiosity_reward,
                    rnd_next_obs=rnd_new_obs
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

            (_, rewems), curiosity_reward_per_env = jax.lax.scan(update_reward_forward_filter, (config["INT_GAMMA"], rewems), transitions.curiosity_reward)

            mean, std, count = (
                jnp.mean(curiosity_reward_per_env),
                jnp.std(curiosity_reward_per_env),
                curiosity_reward_per_env.shape[0]
            )
            reward_rms = update_from_moments(reward_rms, mean, std**2, count)

            metrics["losses"].update({"int_reward": transitions.curiosity_reward.mean()})


            transitions = PPORND2vTransition(
                    obs=transitions.obs,
                    action=transitions.action,
                    reward=transitions.reward,
                    curiosity_reward=transitions.curiosity_reward / jnp.sqrt(reward_rms.var),
                    done=transitions.done,
                    ext_value=transitions.ext_value,
                    int_value=transitions.int_value,
                    log_prob=transitions.log_prob,
                    rnd_next_obs=transitions.rnd_next_obs,
                )

            metrics["time"].update({"env_time": get_time_diff(get_time(), st)})





            # NETWORKS UPDATE
            st = get_time()

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            _, last_ext_val, last_int_val = network.apply(train_state.params, last_obs)

            def _calculate_ext_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.ext_value,
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
                return advantages, advantages + traj_batch.ext_value
            
            def _calculate_int_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.int_value,
                        transition.reward,
                    )
                    done = jnp.zeros_like(done) # Intrinsic target value does not have done flag

                    delta = reward + config["INT_GAMMA"] * next_value * (1 - done) - value
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
                return advantages, advantages + traj_batch.int_value

            ext_advantages, ext_targets = _calculate_ext_gae(transitions, last_ext_val)
            int_advantages, int_targets = _calculate_int_gae(transitions, last_int_val)

            advantages = config["EXT_COEF"] * ext_advantages + config["INT_COEF"] * int_advantages


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
                train_state, traj_batch, advantages, ext_targets, int_targets, rng = carry

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, ext_targets, int_targets = batch_info

                    def _loss_fn(params, traj_batch, gae, ext_targets, int_targets):
                        # RERUN NETWORK
                        pi, ext_value, int_value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        ext_value_pred_clipped = traj_batch.ext_value + (
                            ext_value - traj_batch.ext_value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        ext_value_losses = jnp.square(ext_value - ext_targets)
                        ext_value_losses_clipped = jnp.square(ext_value_pred_clipped - ext_targets)
                        ext_value_loss = (
                            0.5 * jnp.maximum(ext_value_losses, ext_value_losses_clipped).mean()
                        )

                        int_value_loss = 0.5 * (jnp.square(int_value - int_targets)).mean()

                        value_loss = ext_value_loss + int_value_loss

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
                        train_state.params, traj_batch, advantages, ext_targets, int_targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                

                rng, _rng = jax.random.split(rng)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, (traj_batch, advantages, ext_targets, int_targets)
                )

                update_state = (train_state, traj_batch, advantages, ext_targets, int_targets, rng)

                return update_state, total_loss
            



            # Learn RND network 
            def _learn_rnd_epoch(carry, _):
                rnd_train_state, minibatches, rng = carry

                def _learn_phase(carry, minibatch):

                    rnd_train_state, rng = carry

                    def _loss_fn(params):
                        predicted_rnd_features = rnd_network.apply(
                            {"params": params},
                            minibatch.rnd_next_obs,
                        )  # (batch_size*2, num_actions)

                        target_rnd_features = target_rnd_network.apply(
                            {"params": target_rnd_train_state.params},
                            minibatch.rnd_next_obs,
                        )  # (batch_size*2, num_actions)

                        loss = 0.5 * jnp.square(predicted_rnd_features - target_rnd_features).mean()

                        return loss

                    (loss), grads = jax.value_and_grad(
                        _loss_fn
                    )(rnd_train_state.params)
                    rnd_train_state = rnd_train_state.apply_gradients(grads=grads)
                    rnd_train_state = rnd_train_state.replace(
                        grad_steps=rnd_train_state.grad_steps + 1,
                    )
                    return (rnd_train_state, rng), loss

                rng, _rng = jax.random.split(rng)
                (rnd_train_state, rng), loss = jax.lax.scan(
                    _learn_phase, (rnd_train_state, rng), minibatches
                )

                return (rnd_train_state, minibatches, rng), loss



            # Create rng key only once for minibatches, targets and int_targets to maintain consistency
            rng, _rng = jax.random.split(rng)
            minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
            ext_targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), ext_targets
                )
            int_targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), int_targets
                )
            advantages = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), advantages
                )
            
            update_state = (train_state, minibatches, advantages, ext_targets, int_targets, rng)
            

            # Update obs running stats
            obs_rms = update_rms(obs_rms, minibatches.obs[:, -1, :, :].reshape((-1, 1,) + env.observation_space.shape[1:]))
            

            rng, _rng = jax.random.split(rng)
            (train_state, _, _, _, _, rng), loss_info = jax.lax.scan(
                _learn_epoch, update_state, None, config["NUM_EPOCHS"]
            )


            # Learn RND feature function
            # Feature predictor is only updated using a proportion of full experiences
            rnd_minibatches = jax.tree_util.tree_map(
                    lambda x: x[:int(x.shape[0]*config["RND_UPDATE_PROPORTION"])], minibatches
                )  # num_actors*num_envs (batch_size), ...
            rng, _rng = jax.random.split(rng)
            (rnd_train_state, _, rng), rnd_loss = jax.lax.scan(
                _learn_rnd_epoch, 
                (rnd_train_state, rnd_minibatches, rng), 
                None, 
                config["NUM_EPOCHS"]
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
                "rnd_loss": rnd_loss.mean(),
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

            full_train_state = (train_state, rnd_train_state, obs_rms, reward_rms, rewems)
            
            runner_state = (full_train_state, tuple(expl_state), global_metrics, rng)

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
                pi, _, _ = network.apply(train_state.params, last_obs)
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
        



        def _random_walk_and_update_rms(random_runner_state, unused):

            st = get_time()

            train_state, expl_state, global_metrics, obs_rms, rng = random_runner_state

            metrics = {"general": {}, "losses": {}, "time": {}, "misc": {}}
            metrics["misc"].update({"prev_steps": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"]})
            
            def _random_walk(carry, unused):
                last_obs, env_state, rng = carry
                rng, _rng = jax.random.split(rng)

                random_action = jax.random.randint(
                        _rng, shape=(total_envs,), minval=0, maxval=env.single_action_space.n
                )  # sample random actions,

                new_obs, new_env_state, reward, done, info = env.step(
                    env_state, random_action
                )

                return (new_obs, new_env_state, rng), new_obs 

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), tran_obs = jax.lax.scan(
                _random_walk, (*expl_state, _rng), None, config["NUM_STEPS"]
            )
            
            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                tran_obs = tran_obs[:, : -config["TEST_ENVS"]]

            obs_rms = update_rms(obs_rms, tran_obs[:, -1, :, :].reshape((-1, 1,) + env.observation_space.shape[1:]))

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
                # n_updates=train_state.n_updates + 1
            )  # update timesteps count for random walk


            metrics["general"].update({
                "iterations": train_state.n_updates,
                "env_steps": train_state.timesteps,
                "env_frames": train_state.timesteps * config["ENV_KWARGS"]["frame_skip"],
                "grad_steps": train_state.grad_steps,
            })

            metrics["misc"].update({"start_time": global_st})
            metrics["time"].update({"env_time": get_time_diff(get_time(), st)})
            metrics["time"].update({"uptime": get_time_diff(get_time(), global_st)})

            # report statistics
            print_callback(metrics, global_metrics, original_seed, config)

            global_metrics["env_time"] += metrics["time"]["env_time"]
            global_metrics["uptime"] = metrics["time"]["uptime"]

            expl_state = tuple(expl_state)
            random_runner_state = train_state, expl_state, global_metrics, obs_rms, rng

            return random_runner_state, rng
    

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

        random_runner_state, _ = jax.lax.scan(
            _random_walk_and_update_rms, (train_state, (init_obs, env_state), global_metrics, obs_rms, rng), None, config["NUM_RANDOM"]
        )

        train_state, expl_state, global_metrics, obs_rms, rng = random_runner_state


        rng, _rng = jax.random.split(rng)

        full_train_state = (train_state, rnd_train_state, obs_rms, reward_rms, rewems)
        runner_state = (full_train_state, expl_state, global_metrics, _rng)


        if config.get("SAVE_STATE", True):

            ckpt = {"runner_state": train_state, "metrics": global_metrics}
            jax.debug.callback(save_state, ckpt, 0)



        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

