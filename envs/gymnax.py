import jax
import jax.numpy as jnp
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


def make_env(env_name, flatten=True):

    env, env_params = gymnax.make(env_name)
    if flatten:
        env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    return env, env_params, vmap_reset, vmap_step