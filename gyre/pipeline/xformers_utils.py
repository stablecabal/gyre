import functools
import logging

import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

logger = logging.getLogger(__name__)


@functools.cache
def xformers_mea_available():
    available = False

    if is_xformers_available():
        try:
            # Make sure we can run the memory efficient attention
            _ = xformers.ops.memory_efficient_attention(
                torch.randn((1, 2, 40), device="cuda"),
                torch.randn((1, 2, 40), device="cuda"),
                torch.randn((1, 2, 40), device="cuda"),
            )
        except Exception:
            pass
        else:
            available = True

    return available


@functools.cache
def xformers_mea_reversible(size: int):
    @torch.enable_grad()
    def _grad(size):
        q = torch.randn((1, 4, size), device="cuda")
        k = torch.randn((1, 4, size), device="cuda")
        v = torch.randn((1, 4, size), device="cuda")

        q = q.detach().requires_grad_()
        k = k.detach().requires_grad_()
        v = v.detach().requires_grad_()

        out = xformers.ops.memory_efficient_attention(q, k, v)
        loss = out.sum(2).mean(0).sum()

        return torch.autograd.grad(loss, v)

    try:
        _grad(size)
        logger.debug(f"Xformers reverse testing {size} - passed")
        return True
    except Exception as e:
        logger.debug(f"Xformers reverse testing {size} - failed")
        return False


def xformers_mea_reversible_for_module(module: torch.nn.Module):

    if isinstance(module, Attention):
        size = int(module.scale**-2)
        return xformers_mea_reversible(size)

    else:
        raise ValueError(
            "Don't know how to get size for class", module.__class__.__name__
        )
