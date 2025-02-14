import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from functools import partial


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )

        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )