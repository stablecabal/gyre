import torch

from gyre.generated.generation_pb2 import (
    Safetensors,
    SafetensorsMeta,
    SafetensorsTensor,
)
from gyre.protobuf_tensors import deserialize_tensor, serialize_tensor


def serialize_safetensor(safetensors):
    proto_safetensors = Safetensors()

    metadata = safetensors.metadata()
    if metadata:
        for k, v in metadata.items():
            proto_safetensors.metadata.append(SafetensorsMeta(key=k, value=v))

    for k in safetensors.keys():
        proto_safetensors.tensors.append(
            SafetensorsTensor(key=k, tensor=serialize_tensor(safetensors.get_tensor(k)))
        )

    return proto_safetensors


def serialize_safetensor_from_dict(tensors, metadata: dict[str, str] | None = None):
    proto_safetensors = Safetensors()

    for k, v in tensors.items():
        proto_safetensors.tensors.append(
            SafetensorsTensor(key=k, tensor=serialize_tensor(v))
        )

    if metadata:
        for k, v in metadata.items():
            proto_safetensors.metadata.append(SafetensorsMeta(key=k, value=v))

    return proto_safetensors


class UserSafetensors:
    def __init__(
        self, metadata: dict[str, str] = {}, tensors: dict[str, torch.Tensor] = {}
    ):
        self._metadata = metadata
        self._tensors = tensors

    def metadata(self):
        return self._metadata

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, key):
        return self._tensors[key]

    # Extension to Safetensors
    def tensors(self):
        return self._tensors

    def items(self):
        return self._tensors.items()


def deserialize_safetensors(proto_safetensors):
    metadata = {}
    tensors = {}

    for meta in proto_safetensors.metadata:
        metadata[meta.key] = meta.value

    for tensor in proto_safetensors.tensors:
        tensors[tensor.key] = deserialize_tensor(tensor.tensor)

    return UserSafetensors(metadata, tensors)
