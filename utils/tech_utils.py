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