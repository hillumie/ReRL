import jax
import jax.numpy as jnp
import gymnax
# from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from envs.gymnax_wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

def make_env(env_name, gamma, normalize=True):
    env, env_params = BraxGymnaxWrapper(env_name), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if normalize:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)

    return env, env_params