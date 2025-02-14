import utils.debug

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import optax
import chex
from flax.linen.initializers import constant, orthogonal

from functools import partial
from dataclasses import dataclass, field
from typing import Any, List, Dict


class CNN(nn.Module):

    cnn_struct: Any = field(default_factory=[
        {"conv": [32, (8,8), (4,4), "VALID", "he_normal"]},     # [[num_features: int, kernel_size: (int, int), strides: (int, int), padding: str, kernel_init: str]]
        {"norm": "layer_norm"},                                 # layer_norm, batch_norm
        {"acti": "relu"},                                       # relu, tanh
    ])

    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):

        for layer in self.cnn_struct:
            layer_name = next(iter(layer.keys()))

            if layer_name == "conv":
                num_features, kernel_size, strides, padding, kernel_init_name = layer["conv"]
                if kernel_init_name == "he_normal":
                    kernel_init = nn.initializers.he_normal()
                else:
                    raise ValueError("Initialization type is not existed or not yet supported")

                x = nn.Conv(
                    num_features, 
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    kernel_init=kernel_init,
                )(x)

            elif layer_name == "norm":
                norm_type = layer["norm"]
                if norm_type == "layer_norm":
                    normalize = lambda x: nn.LayerNorm()(x)
                elif norm_type == "batch_norm":
                    normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
                elif norm_type == None:
                    normalize = lambda x: x
                else:
                    raise ValueError("Normalization type is not existed or not yet supported")
                x = normalize(x)

            elif layer_name == "acti":
                # TODO: Implement more activation functions
                acti_fn = layer["acti"]
                if acti_fn == "relu":
                    activation_fn = lambda x: nn.relu(x)
                elif acti_fn == "tanh":
                    activation_fn = lambda x: nn.tanh(x)
                else:
                    raise ValueError("Activation function is not existed or not yet supported")
                x = activation_fn(x)

            else:
                raise ValueError("Layer type is not existed or not yet supported")
            
        return x
        