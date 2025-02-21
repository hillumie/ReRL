import skrl
from skrl import logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper
from typing import Any, Mapping, Tuple, Union

try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message
else:
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space

import jax
import jax.numpy as jnp
import gymnasium
import numpy as np

from utils.tech_utils import _torch2jax, _jax2torch


class MSHABWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Maniskill-HAB environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._env_device = torch.device(self._unwrapped.device)
        self._reset_once = True
        self._observations = None
        self._info = {}

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        """State space"""
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        """Observation space"""
        try:
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space["policy"]

    @property
    def action_space(self) -> gymnasium.Space:
        """Action space"""
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(self, actions: Union[np.ndarray, jax.Array]) -> Tuple[
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Union[np.ndarray, jax.Array],
        Any,
    ]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: np.ndarray or jax.Array

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of np.ndarray or jax.Array and any other info
        """
        actions = _jax2torch(actions, self._env_device, self._jax)
        actions = unflatten_tensorized_space(self.action_space, actions)

        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))

        terminated = terminated.to(dtype=torch.int8)
        truncated = truncated.to(dtype=torch.int8)

        self._observations = _torch2jax(observations, self._jax)
        return (
            self._observations,
            _torch2jax(reward.view(-1, 1), self._jax),
            _torch2jax(terminated.view(-1, 1), self._jax),
            _torch2jax(truncated.view(-1, 1), self._jax),
            self._info,
        )

    def reset(self) -> Tuple[Union[np.ndarray, jax.Array], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: np.ndarray or jax.Array and any other info
        """
        if self._reset_once:
            observations, self._info = self._env.reset()
            observations = flatten_tensorized_space(tensorize_space(self.observation_space, observations["policy"]))
            self._observations = _torch2jax(observations, self._jax)
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()