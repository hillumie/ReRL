import numpy as np
import jax.numpy as jnp
from flax import struct
import gymnasium as gym
import jax

from skrl.envs.wrappers.jax.isaaclab_envs import IsaacLabWrapper, IsaacLabMultiAgentWrapper 


from typing import Optional, Sequence

import sys

from skrl import logger


__all__ = ["load_isaaclab_env"]


def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: ``0``)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")


def load_isaaclab_env(
    task_name: str = "",
    num_envs: Optional[int] = None,
    headless: Optional[bool] = None,
    cli_args: Sequence[str] = [],
    show_cfg: bool = True,
):
    import argparse
    import atexit
    import gymnasium

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        arg_index = sys.argv.index("--task") + 1
        if arg_index >= len(sys.argv):
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )
        if task_name and task_name != sys.argv[arg_index]:
            logger.warning(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )

    # check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--num_envs"):
            defined = True
            break
    # get num_envs from command line arguments
    if defined:
        if num_envs is not None:
            logger.warning("Overriding num_envs with command line argument (--num_envs)")
    # get num_envs from function arguments
    elif num_envs is not None and num_envs > 0:
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

    # check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--headless"):
            defined = True
            break
    # get headless from command line arguments
    if defined:
        if headless is not None:
            logger.warning("Overriding headless with command line argument (--headless)")
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append("--headless")

    # others command line arguments
    sys.argv += cli_args

    # parse arguments
    parser = argparse.ArgumentParser("Isaac Lab: Omniverse Robotics Environments!")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )

    # launch the simulation app
    try:
        from omni.isaac.lab.app import AppLauncher
    except ModuleNotFoundError:
        from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app_launcher = AppLauncher(args)

    @atexit.register
    def close_the_simulator():
        app_launcher.app.close()

    try:
        import omni.isaac.lab_tasks  # type: ignore
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore
    except ModuleNotFoundError:
        import isaaclab_tasks  # type: ignore
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)

    # print config
    if show_cfg:
        print(f"\nIsaac Lab environment ({args.task})")
        try:
            _print_cfg(cfg)
        except AttributeError as e:
            pass

    # load environment
    env = gymnasium.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)

    from isaaclab.envs import multi_agent_to_single_agent

    env = multi_agent_to_single_agent(env)

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    env = SkrlVecEnvWrapper(env, ml_framework="jax")
    env = JaxLogEnvPoolWrapper(env)
    
    return env

class IsaacLabEnv():

    def __init__(self, env_name, num_envs=None, video=False, save_video_path=None, single_agent=True, distributed=False, device="cuda", disable_fabric=False):

        self.env = load_isaaclab_env(task_name=env_name, num_envs=num_envs, headless=True)

        # # wrap for video recording
        # if video:
        #     video_kwargs = {
        #         "video_folder": os.path.join(save_video_path, "videos", "train"),
        #         "step_trigger": lambda step: step % self.args_cli.video_interval == 0,
        #         "video_length": self.args_cli.video_length,
        #         "disable_logger": True,
        #     }
        #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # import skrl
        # skrl.config.jax.backend = "jax"
        # self.env = SkrlVecEnvWrapper(env, ml_framework="jax")

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
        print(action)
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