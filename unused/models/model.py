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
from typing import Any, List


class Model(nn.Module):

    model_struct: Any = field(default_factory=[
        {
            "cnn": [
                {"conv": [32, (8,8), (4,4), "VALID", "he_normal"]},    
                {"norm": "layer_norm"},                                 
                {"acti": "relu"},
            ],
            "repeat": 1,          
        }, 
        {
            "reshape": [[0], -1], 
        },
        {
            "mlp": [
                {"dense": [512, "he_normal"]},     
                {"norm": "layer_norm"},           
                {"acti": "relu"},                  
            ],
            "repeat": 1,
        },      
    ])

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        for layer in self.model_struct:
            layer_name = next(iter(layer.keys()))

            if layer_name == "cnn":
                cnn_struct, num_repeat = layer["cnn"], layer["repeat"]
                repeated_cnn_struct = cnn_struct * num_repeat
                
                from models.cnn import CNN
                x = CNN(cnn_struct=repeated_cnn_struct)(x, train)
            
            elif layer_name == "mlp":
                mlp_struct, num_repeat = layer["mlp"], layer["repeat"]
                repeated_mlp_struct = mlp_struct * num_repeat
                
                from models.mlp import MLP
                x = MLP(mlp_struct=repeated_mlp_struct)(x, train)
            
            elif layer_name == "reshape":
                shape_name = layer["reshape"]
                shape = tuple()
                for s in shape_name:
                    if isinstance(s, List) or isinstance(s, tuple):
                        shape += (int(x.shape[s[0]]), )
                    elif isinstance(s, int):
                        shape += (int(s), )
                    else:
                        raise ValueError("Wrong reshape shape type")
                x = x.reshape(shape)
            
            else:
                raise ValueError(f"{layer_name} is not existed or not yet supported")
            
        return x
        