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

from models.model import Model


class Q(nn.Module):

    output_dim: int = 6
    norm_input: bool = False
    model_struct = [
        {
            "cnn": [
                {"conv": [32, (8,8), (4,4), "VALID", "he_normal"]},    
                {"norm": "layer_norm"},                                 
                {"acti": "relu"},
            ],
            "repeat": 1,          
        }, 
        {
            "cnn": [
                {"conv": [64, (4,4), (2,2), "VALID", "he_normal"]},    
                {"norm": "layer_norm"},                                 
                {"acti": "relu"},
            ],
            "repeat": 1,          
        }, 
        {
            "cnn": [
                {"conv": [64, (3,3), (1,1), "VALID", "he_normal"]},    
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
        {
            "mlp": [
                {"dense": [output_dim, "he_normal"]},     
                {"norm": "layer_norm"},           
                {"acti": "relu"},                  
            ],
            "repeat": 1,
        } 
    ]

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x/225.0
        x = Model(model_struct=self.model_struct)(x)
            
        return x
        