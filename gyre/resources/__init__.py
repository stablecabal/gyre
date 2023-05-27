import io
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import huggingface_hub
import torch
from huggingface_hub.file_download import http_get
from safetensors import safe_open
from safetensors.torch import load as safe_load_bytes

from gyre import torch_safe_unpickler
from gyre.cache import MB, CacheLookupError, TensorLRUCache_LockBase
from gyre.protobuf_safetensors import UserSafetensors

# from gyre.resources.civitai import Civitai

SOURCES = {"local" "civitai", "huggingface", "web"}
TYPES = {"lora", "embedding", "image"}

DEFAULT_WHITELIST = [
    dict(source="local", type="*", format="*"),
    dict(source="civitai", type="lora", format="safetensors", max_size=384 * MB),
    dict(source="civitai", type="embedding", format="safetensors", max_size=8 * MB),
    dict(source="huggingface", type="lora", format="safetensors", max_size=384 * MB),
    dict(source="huggingface", type="embedding", format="safetensors", max_size=8 * MB),
]


class ResourcePermissionError(RuntimeError):
    pass


class ResourceProvider:
    def __init__(self, cache, whitelist=DEFAULT_WHITELIST):
        self.cache: TensorLRUCache_LockBase = cache
        self.whitelist = whitelist
        self.locals = []

    def add_whitelist_line(self, line: dict):
        self.whitelist.append(line)

    def _split_path(self, path):
        parts = path.split("/")
        parts = [part for part in parts if part and part != "." and part != ".."]
        return parts

    def add_local_path(self, url_prefix, path, restype="*"):
        if not os.path.isabs(path):
            raise ValueError(f"path must be absolute, but got {path}")

        self.locals.append(
            dict(prefix=self._split_path(url_prefix), path=path, restype=restype)
        )
        self.locals.sort(key=lambda x: len(x["prefix"]), reverse=True)

    # A check will fail if there is a requirement that we don't know is met or not
    # (not included in kwargs). A precheck will only fail if there is a requirement
    # that we know isn't met (included in kwargs but doesn't match)

    def _check_line(self, line, **kwargs):
        for k, v in line.items():
            if v == "*":
                continue
            if k == "max_size":
                if "size" not in kwargs or kwargs["size"] > v:
                    return False
            else:
                if k not in kwargs or kwargs[k] != v:
                    return False
        return True

    def _precheck_line(self, line, **kwargs):
        for k, v in line.items():
            if v == "*":
                continue
            if k == "max_size":
                if "size" in kwargs and kwargs["size"] > v:
                    return False
            else:
                if k in kwargs and kwargs[k] != v:
                    return False
        return True

    def _check_whilelist(self, **kwargs):
        if any((self._check_line(line, **kwargs) for line in self.whitelist)):
            return True

        raise ResourcePermissionError(
            f"URL with parameters {kwargs} failed whitelist check."
        )

    def _precheck_whilelist(self, **kwargs):
        if any((self._precheck_line(line, **kwargs) for line in self.whitelist)):
            return True

        raise ResourcePermissionError(
            f"URL with parameters {kwargs} failed whitelist precheck."
        )

    def get(self, restype, url):
        parts = urlparse(url)
        key = f"resource-{restype}-{url}"

        if parts.scheme != "file" and parts.fragment != "refresh":
            # If it's in the cache, it doesn't need to be whitelisted
            try:
                return self.cache.get_safetensors(key)
            except CacheLookupError:
                pass

        self._precheck_whilelist(type=restype)

        if parts.scheme == "file":
            metadata, tensors = self._get_file(restype, parts.netloc + parts.path)

        elif parts.netloc == "civitai.com":
            metadata, tensors = self._get_civitai(restype, parts.path)

        elif parts.netloc == "huggingface.co":
            metadata, tensors = self._get_huggingface(restype, parts.path)

        elif parts.scheme == "https":
            metadata, tensors = self._get_web(restype, parts)

        else:
            raise ResourcePermissionError(f"Unhandled resource URL '{url}'")

        self.cache.set(key, metadata=metadata, tensors=tensors)
        return UserSafetensors(metadata=metadata, tensors=tensors)

    def _parse_safetensor_metadata(self, byt):
        header_len = int.from_bytes(byt[:8], "little")
        metadata = json.loads(byt[8 : 8 + header_len])
        return metadata.get("__metadata__", {})

    def _is_torch_statefile(self, byt):
        return byt[:4] == b"PK\x03\x04"

    def _deserialise(self, restype, fp, byt):
        if self._is_torch_statefile(byt):
            fp.seek(0)
            metadata = {}
            tensors = torch.load(
                fp, map_location="cpu", pickle_module=torch_safe_unpickler
            )

            # Bit of a hack, token embeddings often have the tensors in a nested key,
            # which we need to pull out
            if restype == "embedding" and "string_to_param" in tensors:
                tensors = tensors["string_to_param"]

        else:
            metadata = self._parse_safetensor_metadata(byt)
            tensors = safe_load_bytes(byt)

        return metadata, tensors

    def _get_file_path(self, restype, path):
        parts = self._split_path(path)

        for candidate in self.locals:
            prefix_match = candidate["prefix"] == parts[: len(candidate["prefix"])]
            restype_match = candidate["restype"] in {"*", restype}

            if prefix_match and restype_match:
                candidate_path = (
                    candidate["path"]
                    + "/"
                    + "/".join(parts[len(candidate["prefix"]) :])
                )

                if Path(candidate_path).exists():
                    return candidate_path

        raise ResourcePermissionError(f"No local path configured for {restype} {path}")

    def _get_file(self, restype, path):
        self._precheck_whilelist(source="local", type=restype)

        file_path = self._get_file_path(restype, path)

        fp = open(file_path, "rb")
        byt = fp.read()

        frmt = "pt" if self._is_torch_statefile(byt) else "safetensors"
        self._check_whilelist(source="local", type=restype, size=len(byt), format=frmt)

        return self._deserialise(restype, fp, byt)

    def _download(self, restype, url, source):
        fp = io.BytesIO()
        http_get(url, fp)
        byt = fp.getvalue()

        frmt = "pt" if self._is_torch_statefile(byt) else "safetensors"
        self._check_whilelist(source=source, type=restype, size=len(byt), format=frmt)

        return self._deserialise(restype, fp, byt)

    def _get_civitai(self, restype, path):
        self._precheck_whilelist(source="civitai", type=restype)
        return self._download(restype, "https://civitai.com" + path, "civitai")

    def _get_huggingface(self, restype, path):
        self._precheck_whilelist(source="huggingface", type=restype)

        parts = self._split_path(path)

        if "blob" not in parts and "resolve" not in parts:
            raise ResourcePermissionError(
                f"Can't understand huggingface path {path}, "
                f"should have 'blob' or 'resolve' somewhere but doesn't"
            )

        parts = [("resolve" if part == "blob" else part) for part in parts]

        return self._download(
            restype, "https://huggingface.co/" + "/".join(parts), source="huggingface"
        )

    def _get_web(self, restype, path):
        self._precheck_whilelist(source="web", type=restype)
