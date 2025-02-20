import argparse

import numpy as np
import jax.numpy as jnp
from flax import struct
import gymnasium as gym
import jax

from skrl.envs.wrappers.jax.isaaclab_envs import IsaacLabWrapper, IsaacLabMultiAgentWrapper 

from isaaclab.app import AppLauncher

class IsaacLabEnv():

    def __init__(self, env_name, num_envs=None, video=False, save_video_path=None, single_agent=True, distributed=False, device="cuda", disable_fabric=False):

        # add argparse arguments
        parser = argparse.ArgumentParser(description="IsaacLab Envionment")

        AppLauncher.add_app_launcher_args(parser)
        self.args_cli, self.hydra_args = parser.parse_known_args()
        self.args_cli.headless = True
        # always enable cameras to record video
        if video:
            self.args_cli.enable_cameras = True

        # launch omniverse app
        self.app_launcher = AppLauncher(self.args_cli)
        simulation_app = self.app_launcher.app


        import gymnasium as gym
        import os

        from isaaclab.envs import (
            DirectMARLEnv,
            DirectMARLEnvCfg,
            DirectRLEnvCfg,
            ManagerBasedRLEnvCfg,
            multi_agent_to_single_agent,
        )
        from isaaclab.utils.dict import print_dict
        from isaaclab_tasks.utils import parse_env_cfg

        import isaaclab_tasks  # noqa: F401

        # parse configuration
        env_cfg = parse_env_cfg(
            env_name, device=device, num_envs=num_envs, use_fabric=not disable_fabric
        )

        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = num_envs if num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = device if device is not None else env_cfg.sim.device

        # multi-gpu training config
        if distributed:
            env_cfg.sim.device = f"cuda:{self.app_launcher.local_rank}"
        
        env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array" if video else None)
        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv) and single_agent:
            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if video:
            video_kwargs = {
                "video_folder": os.path.join(save_video_path, "videos", "train"),
                "step_trigger": lambda step: step % self.args_cli.video_interval == 0,
                "video_length": self.args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        import skrl
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        skrl.config.jax.backend = "jax"
        self.env = SkrlVecEnvWrapper(env, ml_framework="jax")
        self.env = JaxLogEnvPoolWrapper(self.env)

    def make_env(self):
        return self.env

@struct.dataclass
class LogEnvState:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class JaxLogEnvPoolWrapper(IsaacLabWrapper):

    def __init__(self, env, reset_info=True, async_mode=True):
        super(JaxLogEnvPoolWrapper, self).__init__(env)
        self.env = env

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)

        env_state = LogEnvState(
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observations, env_state

    def step(self, state, action):

        state = jax.device_put(state, device=jax.sharding.SingleDeviceSharding(jax.devices()[0]))
        # action = jax.device_put(action, device=jax.sharding.SingleDeviceSharding(jax.devices()[0]))

        observations, rewards, terminated, truncated, infos_ = self.env.step(
            action
        )

        rewards = rewards.squeeze()
        terminated = terminated.squeeze()
        truncated = truncated.squeeze()
    
        new_episode_return = state.episode_returns + rewards
        new_episode_length = state.episode_lengths + 1
        state = state.replace(
            episode_returns=(new_episode_return)
            * (1 - terminated)
            * (1 - truncated),
            episode_lengths=(new_episode_length)
            * (1 - terminated)
            * (1 - truncated),
            returned_episode_returns=jnp.where(
                terminated + truncated,
                new_episode_return,
                state.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                terminated + truncated,
                new_episode_length,
                state.returned_episode_lengths,
            ),
        )

        # if self.reset_info:
        #     elapsed_steps = infos["elapsed_step"]
        #     terminated = infos["terminated"] + infos["TimeLimit.truncated"]
        
        infos = {}
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["returned_episode"] = terminated

        return (
            observations,
            state,
            rewards,
            terminated,
            truncated,
            infos,
        )