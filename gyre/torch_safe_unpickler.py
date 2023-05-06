# Initially based on https://github.com/pytorch/pytorch/blob/v1.13.1/torch/_weights_only_unpickler.py
# but adjusted to use PickleMagic to unpickle, allowing unpickling of pickles with
# unsafe or unknown types in them

import functools as _functools
import logging
from collections import OrderedDict
from typing import Any, Dict, List

import torch
from picklemagic import FakeUnpickler

logger = logging.getLogger(__name__)

# Unpickling machinery
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    rc: Dict[str, Any] = {
        "collections.OrderedDict": OrderedDict,
        "torch.nn.parameter.Parameter": torch.nn.Parameter,
        "torch.serialization._get_layout": torch.serialization._get_layout,
        "torch.Size": torch.Size,
        "torch.Tensor": torch.Tensor,
    }
    # dtype
    for t in [
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        rc[str(t)] = t
    # Tensor classes
    for tt in torch._tensor_classes:
        rc[f"{tt.__module__}.{tt.__name__}"] = tt
    # Storage classes
    for ts in torch._storage_classes:
        rc[f"{ts.__module__}.{ts.__name__}"] = ts
    # Rebuild functions
    for f in [
        torch._utils._rebuild_parameter,
        torch._utils._rebuild_tensor,
        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_sparse_tensor,
        torch._utils._rebuild_meta_tensor_no_storage,
        torch._utils._rebuild_sparse_csr_tensor,
    ]:
        rc[f"torch._utils.{f.__name__}"] = f
    return rc


class Unpickler(FakeUnpickler):
    def __init__(self, file, *, encoding: str = "bytes"):
        super().__init__(file, encoding=encoding)

    def find_class(self, module, name):
        safe = _get_allowed_globals()
        fqn = f"{module}.{name}"

        if fqn in safe:
            return safe[fqn]

        # Logging for now
        logger.warn(f"skipping unpickling {fqn} as unsafe.")
        return self.class_factory(name, module)

    def get_extension(self, code):
        logger.warn("skipping unpickling extension {code} as unsafe.")
        return self.class_factory("extension_code_{0}".format(code), "copyreg")


def load(file, *, encoding: str = "ASCII"):
    return Unpickler(file, encoding=encoding).load()
