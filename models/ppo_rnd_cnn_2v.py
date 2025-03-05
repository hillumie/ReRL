import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0

        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(448, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)

        actor_mean = nn.Dense(
            448, kernel_init=nn.initializers.he_normal()
        )(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.he_normal()
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)


        x_critic = nn.Dense(448, kernel_init=nn.initializers.he_normal())(x)
        ext_critic = nn.Dense(1, kernel_init=nn.initializers.he_normal())(
            x_critic + x
        )
        int_critic = nn.Dense(1, kernel_init=nn.initializers.he_normal())(
            x_critic + x
        )

        return pi, jnp.squeeze(ext_critic, axis=-1), jnp.squeeze(int_critic, axis=-1)
    


class RNDCNN(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        return x
    

class TargetFeatureRNDCNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train=False):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = RNDCNN()(x)
        return x
    
class FeatureRNDCNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train=False):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = RNDCNN()(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        return x
    