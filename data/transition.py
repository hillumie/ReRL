import jax
import jax.numpy as jnp
import chex
from typing import Sequence, NamedTuple, Any

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    next_obs: chex.Array
    reward: chex.Array
    done: chex.Array
    q_val: chex.Array
    int_q_val: chex.Array = jnp.array([0])
    curiosity_reward: chex.Array = jnp.array([0])
    rnd_next_obs: chex.Array = jnp.array([0])


@chex.dataclass(frozen=True)
class RNNTransition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array


class ContinuousTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray