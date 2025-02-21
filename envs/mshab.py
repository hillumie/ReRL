import jax
import jax.numpy as jnp
from flax import struct
import gymnasium as gym
from envs.non_jax_wrappers import MSHABWrapper

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mshab.envs
from mshab.envs.planner import plan_data_from_file


class MSHABEnv():
    def __init__(self, env_name, num_envs=None, seed=42):

        task, subtask, split = env_name.split("-")
        task = "tidy_house" # "tidy_house", "prepare_groceries", or "set_table"
        subtask = "pick"    # "sequential", "pick", "place", "open", "close"
                            # NOTE: sequential loads the full task, e.g pick -> place -> ...
                            #     while pick, place, etc only simulate a single subtask each episode
        split = "train"     # "train", "val"


        REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

        plan_data = plan_data_from_file(
            REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
        )
        spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

        env = gym.make(
            f"{subtask.capitalize()}SubtaskTrain-v0",
            # Simulation args
            num_envs=num_envs,     # 252,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
            obs_mode="rgbd",
            sim_backend="gpu",
            robot_uids="fetch",
            control_mode="pd_joint_delta_pos",
            # Rendering args
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            shader_dir="minimal",
            # TimeLimit args
            max_episode_steps=200,
            # SequentialTask args
            task_plans=plan_data.plans,
            scene_builder_cls=plan_data.dataset,
            # SubtaskTrain args
            spawn_data_fp=spawn_data_fp,
            # optional: additional env_kwargs
        )

        # add env wrappers here

        env = ManiSkillVectorEnv(
            env,
            max_episode_steps=1000,  # set manually based on task
            ignore_terminations=True,  # set to False for partial resets
        )

        self.env = MSHABWrapper(env)

    
    def make_env(self):
        return self.env

@struct.dataclass
class LogEnvState:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class JaxLogEnvPoolWrapper(MSHABWrapper):

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