import chex
import jax.numpy as jnp
from typing import Sequence, NamedTuple, Any
from functools import partial

import jax
import jax.dlpack as jax_dlpack

try:
    import torch
    import torch.utils.dlpack as torch_dlpack
except:
    pass  # TODO: show warning message

_CPU = jax.devices()[0].device_kind.lower() == "cpu"

def _jax2torch(array, device, from_jax=True):
    if from_jax:
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array)).to(device=device)
    return torch.tensor(array, device=device)


def _torch2jax(tensor, to_jax=True):
    if to_jax:
        return jax_dlpack.from_dlpack(
            torch_dlpack.to_dlpack(tensor.contiguous().cpu() if _CPU else tensor.contiguous())
        )
    return tensor.cpu().numpy()




@chex.dataclass(frozen=True)
class RunningMeanStd:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.float32

def update_rms(stats: RunningMeanStd, x):
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]
    stats = update_from_moments(stats, batch_mean, batch_var, batch_count)

    return stats

def update_from_moments(stats: RunningMeanStd, batch_mean, batch_var, batch_count):
    new_mean, new_var, new_count = update_mean_var_count_from_moments(
            stats.mean, stats.var, stats.count, batch_mean, batch_var, batch_count
        )
    stats = RunningMeanStd(
        mean = new_mean,
        var = new_var,
        count = new_count
    )

    return stats

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count



@partial(jax.jit)
def update_reward_forward_filter(carry, reward):
    gamma, rewems = carry
    rewems = rewems * gamma + reward
    return (gamma, rewems), rewems