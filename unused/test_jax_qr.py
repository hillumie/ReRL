import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

def init_params(key):
    # Initialize on CPU for testing
    key, subkey = random.split(key)
    shape = (64, 64)  # Example shape
    # Use QR decomposition for orthogonal initialization
    q, _ = jnp.linalg.qr(random.normal(subkey, shape))
    return q

# Test on CPU
key = random.PRNGKey(0)
params = init_params(key)