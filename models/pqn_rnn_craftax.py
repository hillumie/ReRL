import jax
import jax.numpy as jnp
import flax.linen as nn
from models.batch_renorm import BatchRenorm
from models.scanned_rnn import ScannedRNN



class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False
    add_last_action: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        # add last action to the input of the rnn
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)

        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        q_vals = nn.Dense(self.action_dim)(x)

        return new_hidden, q_vals

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]