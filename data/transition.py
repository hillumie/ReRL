import jax
import jax.numpy as np
import chex

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    next_obs: chex.Array
    reward: chex.Array
    done: chex.Array
    q_val: chex.Array


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