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

from models.ppo_mlp_isaaclab import ContinuousActorCritic
from data.transition import ContinuousTransition 
from utils.common_utils import log_callback, save_state, print_dashboard, get_time, get_time_diff

from envs.isaaclab import IsaacLabEnv

class CustomTrainState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


# Impure agent to use impure env but does not lose much speed on training
class PPOIsaacAgent():

    def __init__(self, config, rng):

        self.config = config

        self.config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )

        config["TOTAL_TIMESTEPS_DECAY"] = config["TOTAL_TIMESTEPS"]
        self.config["NUM_UPDATES_DECAY"] = (
            config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )

        self.config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )

        self.env = IsaacLabEnv(config["ENV_NAME"], config["NUM_ENVS"]).make_env()
        
        # self.eval_env = IsaacLabEnv(config["ENV_NAME"], config["TEST_NUM_ENVS"]).make_env()

        self.config["TEST_NUM_STEPS"] = config.get(
            "TEST_NUM_STEPS", 1,
        )


        self.lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["UPDATE_EPOCHS"],
        )
        self.lr = self.lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        self.network = ContinuousActorCritic(
            self.env.action_space.shape[0], 
            activation=config["ACTIVATION"],
        )

        self.original_seed = rng[0]
        self.original_rng = rng
        self.train_state = self.create_agent(rng)

        self.global_st = get_time()


    def create_agent(self, rng):

        init_x = jnp.zeros(self.env.observation_space.shape)
        network_params = self.network.init(rng, init_x)
        # tabulate_fn = nn.tabulate(self.network, jax.random.PRNGKey(0))
        # print(tabulate_fn(init_x))

        if self.config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )
        train_state = CustomTrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
        )

        return train_state


    # TRAINING LOOP
    @partial(jax.jit, static_argnames=["self"])
    def _update_step(self, runner_state, unused): 

        train_state, expl_state, global_metrics, rng = runner_state

        metrics = {"general": {}, "losses": {}, "time": {}, "misc": {}}

        metrics["misc"].update({"prev_steps": train_state.timesteps * self.config["ENV_KWARGS"]["frame_skip"]})




        # SAMPLE PHASE
        st = get_time()

        def _env_step(carry, unused):
            last_obs, env_state, rng = carry

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self.network.apply(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            # rng_step = jax.random.split(_rng, self.config["NUM_ENVS"])
            obs, env_state, reward, ter, trun, info = jax.experimental.io_callback(
                self.env.step, 
                (self.fake_obs, self.fake_env_state, self.fake_reward, self.fake_ter, self.fake_trun, self.fake_info), 
                env_state, action,
            )
            done = jnp.logical_or(ter, trun)
            transition = ContinuousTransition(
                done, action, value, reward, log_prob, last_obs
            )

            return (obs, env_state, rng), (transition, info)

        # step the env
        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), (traj_batch, infos) = jax.lax.scan(
            _env_step, 
            (*expl_state, _rng), 
            None, 
            self.config["NUM_STEPS"]
        )
    
        expl_state = tuple(expl_state)

        metrics["time"].update({"env_time": get_time_diff(get_time(), st)})





        # NETWORKS UPDATE
        st = get_time()

        train_state = train_state.replace(
            timesteps=train_state.timesteps
            + self.config["NUM_STEPS"] * self.config["NUM_ENVS"]
        )  # update timesteps count

        # CALCULATE ADVANTAGE
        last_obs, env_state = expl_state

        _, last_val = self.network.apply(train_state.params, last_obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
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

        advantages, targets = _calculate_gae(traj_batch, last_val)


        # UPDATE NETWORK

        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = self.network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
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
                            1.0 - self.config["CLIP_EPS"],
                            1.0 + self.config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + self.config["VF_COEF"] * value_loss
                        - self.config["ENT_COEF"] * entropy
                    )
                    
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = self.config["MINIBATCH_SIZE"] * self.config["NUM_MINIBATCHES"]
            assert (
                batch_size == self.config["NUM_STEPS"] * self.config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"

            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [self.config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss
    
        update_state = (train_state, traj_batch, advantages, targets, rng)
        
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, self.config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        rng = update_state[-1]

        train_state = train_state.replace(n_updates=train_state.n_updates + 1)

        metrics["time"].update({"train_time": get_time_diff(get_time(), st)})




        # EVALUATE AGENT
        st = get_time()
        
        eval_metrics = jax.lax.cond(jnp.logical_or(train_state.n_updates % int(self.config["NUM_UPDATES"]*self.config["EVAL_INTERVAL"]) == 0, 
                                                    train_state.n_updates == self.config["NUM_UPDATES"]),
                                    lambda _: self.eval_agent(train_state, _rng), lambda _: global_metrics["eval"], operand=None)                          
        
        global_metrics["eval"] = eval_metrics

        metrics["time"].update({"eval_time": get_time_diff(get_time(), st)})





        # REPORT METRICS
        metrics["general"].update({
            "iterations": train_state.n_updates,
            "env_steps": train_state.timesteps,
            "env_frames": train_state.timesteps,
            "grad_steps": train_state.grad_steps,
        })

        total_loss, (value_loss, loss_actor, entropy) = loss_info
        metrics["losses"].update({
            "total_loss": total_loss.mean(),
            "value_loss": value_loss.mean(),
            "actor_loss": loss_actor.mean(),
            "entropy": entropy.mean(),
        })
        # metrics["losses"].update({k: v.mean() for k, v in loss_info.items()})

        metrics["general"].update({k: v.mean() for k, v in infos.items()})

        metrics["time"].update({"uptime": get_time_diff(get_time(), self.global_st)})
        metrics["misc"].update({"start_time": self.global_st})

        # report statistics
        log_callback(metrics, global_metrics, self.original_seed, self.config)

        global_metrics["env_time"] += metrics["time"]["env_time"]
        global_metrics["eval_time"] += metrics["time"]["eval_time"]
        global_metrics["train_time"] += metrics["time"]["train_time"]
        global_metrics["uptime"] = metrics["time"]["uptime"]

        runner_state = (train_state, tuple(expl_state), global_metrics, rng)

        if self.config.get("SAVE_STATE", True):
            
            ckpt = {"runner_state": train_state, "metrics": metrics}
            def save_callback(ckpt, iteration):
                jax.debug.callback(save_state, ckpt, iteration)
                return
            def save_nothing(ckpt, iteration):
                return

            jax.lax.cond(jnp.logical_or(train_state.n_updates % int(self.config["NUM_UPDATES"]*self.config["SAVE_INTERVAL"]) == 0, 
                                        train_state.n_updates == self.config["NUM_UPDATES"]), 
                            save_callback, save_nothing, ckpt, metrics["general"]["iterations"])

        return runner_state, metrics

    @partial(jax.jit, static_argnames=["self"])
    def eval_agent(self, train_state, rng):

        # EVAL PHASE
        if not self.config.get("TEST_DURING_TRAINING", False):
            return None


        def _env_step(carry, _):
            last_obs, env_state, rng = carry

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = self.network.apply(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, self.config["TEST_NUM_ENVS"])

            obs, env_state, reward, ter, trun, info = jax.experimental.io_callback(
                self.env.step, 
                (self.fake_obs, self.fake_env_state, self.fake_reward, self.fake_ter, self.fake_trun, self.fake_info), 
                env_state, action,
            )

            # This is not tracable -> model foward time is slow -> slow
            # obs, env_state, reward, ter, trun, info = self.env.step(env_state, action)

            return (obs, env_state, rng), info


        # step the env

        # reset_rng = jax.random.split(rng, self.config["TEST_NUM_ENVS"])
        init_obs, env_state = self.env.reset()

        # TODO: TEST_NUM_STEPS should not be used here, instead loop until done like pqn_atari_agent
        rng, _rng = jax.random.split(rng)
        _, infos = jax.lax.scan(
            _env_step, 
            (init_obs, env_state, rng), 
            None, 
            self.config["TEST_NUM_STEPS"]
        )
    
        # return mean of done infos
        done_infos = jax.tree_map(
            lambda x: jnp.nanmean(
                jnp.where(
                    infos["returned_episode"],
                    x,
                    jnp.nan,
                )
            ),
            infos,
        )

        done_infos = {f"eval_{k}": v for k, v in done_infos.items()}

        return done_infos
    
    def run(self):
        # train
        global_metrics = dict(
            env_time = 0.0,
            eval_time = 0.0,
            train_time = 0.0,
            uptime = 0.0
        )

        # Create fake transition for callback argument
        init_obs, env_state = self.env.reset()
        rng, _rng = jax.random.split(self.original_rng)
        pi, _ = self.network.apply(self.train_state.params, init_obs)
        action = pi.sample(seed=_rng)
        self.fake_obs, self.fake_env_state, self.fake_reward, self.fake_ter, self.fake_trun, self.fake_info = self.env.step(env_state, action)

        rng, _rng = jax.random.split(self.original_rng)
        eval_metrics = self.eval_agent(self.train_state, _rng)
        global_metrics.update({"eval": eval_metrics})


        rng, _rng = jax.random.split(rng)
        # reset_rng = jax.random.split(_rng, self.config["NUM_ENVS"])
        expl_state = self.env.reset()

        runner_state = (self.train_state, expl_state, global_metrics, _rng)


        if self.config.get("SAVE_STATE", True):
                
            ckpt = {"runner_state": self.train_state, "metrics": global_metrics}
            jax.debug.callback(save_state, ckpt, 0)

        # Avoid using Python impure for loop to increase speed (Cartpole SPS: 70k -> 63k)
        # for _ in range(int(self.config["NUM_UPDATES"])):
        #     runner_state, metrics = self._update_step(runner_state, None)

        runner_state, metrics = jax.lax.scan(
            self._update_step, runner_state, None, self.config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

