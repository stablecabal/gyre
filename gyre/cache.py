import hashlib
import logging
import os
import sqlite3
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import safetensors
import torch
from safetensors import torch as safe_torch

from gyre.constants import GB, KB, MB, sd_cache_home
from gyre.protobuf_safetensors import UserSafetensors

logger = logging.getLogger(__name__)

# The debugging info is a bit to detailed for normal use.
logger.setLevel(logging.INFO)

NOT_PASSED_MARKER = object()


@dataclass
class CacheDetails:
    key: str
    access_ctr: int
    size: int
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    expires: float = -1
    path: str | None = None

    def params(self):
        return {k: getattr(self, k) for k in {"key", "tensors", "metadata", "expires"}}

    @property
    def has_expired(self):
        return self.expires > 0 and self.expires < time.time()


class CacheLookupError(RuntimeError):
    pass


class CacheKeyError(CacheLookupError):
    pass


class CacheExpiredError(CacheLookupError):
    pass


TensorMetadataTuple = tuple[dict[str, torch.Tensor], dict[str, str]]
TensorMetadataExpiresTuple = tuple[dict[str, torch.Tensor], dict[str, str], float]


class TensorLRUCache_LockBase:
    def __init__(self, lock=None):
        self.lock = lock or threading.Lock()

    def get(self, key: str) -> TensorMetadataTuple:
        with self.lock:
            tensors, metadata, _ = self._get(key)
            return tensors, metadata

    def get_with_expires(self, key: str) -> TensorMetadataExpiresTuple:
        return self._get(key)

    def get_safetensors(self, key: str) -> UserSafetensors:
        with self.lock:
            tensors, metadata, _ = self._get(key)
            return UserSafetensors(tensors=tensors, metadata=metadata)

    def set(self, key, tensors, metadata=None, expires=None, max_age=None):
        with self.lock:
            return self._set(
                key, tensors, metadata=metadata, expires=expires, max_age=max_age
            )

    def set_safetensors(self, key, safetensors, *args, **kwargs):
        if hasattr(safetensors, "tensors"):
            tensors = safetensors.tensors()
        else:
            tensors = {k: safetensors.get_tensor(k) for k in safetensors.keys()}

        self.set(
            key=key, tensors=tensors, metadata=safetensors.metadata(), *args, **kwargs
        )

    def keyspace(self, key_prefix):
        return TensorLRUCache_Keyspace(key_prefix, self)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, tensors):
        self.set(key, tensors)

    def _get(self, key: str) -> TensorMetadataExpiresTuple:
        raise NotImplementedError()

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None):
        raise NotImplementedError()

    def __contains__(self, key):
        raise NotImplementedError()


class TensorLRUCache_Mem(TensorLRUCache_LockBase):
    def __init__(self, limit, lock=None, on_evict=None):
        super().__init__(lock=lock)

        self.limit: int = limit
        self.cache: dict[str, CacheDetails] = {}
        self.access_ctr: int = 0

        self.on_evict = on_evict

    @property
    def next_ctr(self):
        self.access_ctr += 1
        return self.access_ctr

    def _size_of(self, tensors):
        return sum((t.nelement() * t.element_size() for t in tensors.values()))

    def __contains__(self, key):
        return key in self.cache

    def _get(self, key: str) -> TensorMetadataExpiresTuple:
        if key in self.cache:
            expires = self.cache[key].expires

            if self.cache[key].has_expired:
                raise CacheExpiredError(f"{expires} has passed")

            self.cache[key].access_ctr = self.next_ctr
            return self.cache[key].tensors, self.cache[key].metadata, expires

        raise CacheKeyError(key)

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None):
        if expires is None:
            expires = time.time() + max_age if max_age else -1

        self.cache[key] = CacheDetails(
            key=key,
            access_ctr=self.next_ctr,
            size=self._size_of(tensors),
            tensors=tensors,
            metadata={} if metadata is None else metadata,
            expires=expires,
        )

        self._evict()

    def _needs_evicting(self):
        items = self.cache.values()
        items = sorted(items, key=lambda entry: entry.access_ctr)

        total = 0
        while items and total < self.limit:
            total += items.pop().size

        if items:
            logger.debug(
                f"{total} bytes in mem cache, {len(items)} too many items, evicting"
            )
        else:
            logger.debug(f"{total} bytes in mem cache")

        yield from items

    def _evict(self):
        for item in self._needs_evicting():
            if self.on_evict is not None and not item.has_expired:
                self.on_evict(item)

            logger.debug(f"Evicting {item.key}")
            del self.cache[item.key]


class TensorLRUCache_Disk(TensorLRUCache_LockBase):
    def __init__(self, limit, basepath, lock=None):
        super().__init__(lock=lock)
        self.limit: int = limit
        self.basepath: Path = Path(basepath)
        self.basepath.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache disk folder: {self.basepath}")

    def _path(self, key):
        hashedkey = hashlib.sha256(key.encode("utf-8")).hexdigest().lower()
        return self.basepath / (hashedkey + ".safetensors")

    def __contains__(self, key):
        return self._path(key).exists()

    def _get(self, key) -> TensorMetadataExpiresTuple:
        if (path := self._path(key)).exists():
            safedata = safetensors.safe_open(str(path), framework="pt", device="cpu")
            metadata = safedata.metadata()
            tensors = {k: safedata.get_tensor(k) for k in safedata.keys()}

            expires = float(metadata.pop("__expires", -1))
            if expires > 0 and expires < time.time():
                raise CacheExpiredError(f"{expires} has passed")

            path.touch(exist_ok=True)
            return tensors, metadata, expires

        raise CacheKeyError(key)

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None):
        if expires is None:
            expires = time.time() + max_age if max_age else -1

        if metadata is None:
            metadata = {}

        path = self._path(key)
        safe_torch.save_file(tensors, str(path), metadata | {"__expires": str(expires)})
        self._evict()

    def _needs_evicting(self):
        items = [(st, st.stat()) for st in self.basepath.glob("*.safetensors")]
        items = sorted(items, key=lambda entry: entry[1].st_mtime)

        total = 0
        while items and total < self.limit:
            total += items.pop()[1].st_size

        if items:
            logger.debug(
                f"{total} bytes in disk cache, {len(items)} too many items, evicting"
            )
        else:
            logger.debug(f"{total} bytes in disk cache")

        yield from (item[0] for item in items)

    def _evict(self):
        skipped = 0
        for item in self._needs_evicting():
            try:
                os.remove(item)
            except IOError:
                skipped += 1

        if skipped:
            logger.info(f"Couldn't remove {skipped} item(s) from cache")


class TensorLRUCache_Dual(TensorLRUCache_LockBase):
    def __init__(self, basepath, memlimit=1 * GB, disklimit=10 * GB):
        super().__init__(threading.RLock())
        self.disk_cache = TensorLRUCache_Disk(disklimit, basepath, lock=self.lock)
        self.mem_cache = TensorLRUCache_Mem(memlimit, lock=self.lock)

    def _get(self, key) -> TensorMetadataExpiresTuple:
        try:
            return self.mem_cache.get_with_expires(key)
        except CacheLookupError:
            pass

        try:
            tensor, metadata, expires = self.disk_cache.get_with_expires(key)
            self.mem_cache.set(key, tensor, metadata=metadata, expires=expires)
            return tensor, metadata, expires
        except CacheLookupError:
            pass

        raise CacheLookupError(key)

    def _set(self, *args, **kwargs):
        self.mem_cache.set(*args, **kwargs)
        self.disk_cache.set(*args, **kwargs)

    def __contains__(self, key):
        return key in self.mem_cache or key in self.disk_cache


class TensorLRUCache_Spillover(TensorLRUCache_Dual):
    def __init__(self, basepath=None, memlimit=1 * GB, disklimit=10 * GB):
        super().__init__(basepath=basepath, memlimit=memlimit, disklimit=disklimit)
        self.mem_cache.on_evict = self._on_mem_evict

    def _set(self, *args, **kwargs):
        # Unlike Dual, only set in memory cache
        self.mem_cache.set(*args, **kwargs)

    def _on_mem_evict(self, item):
        # If expired from memory cache, spill over to disk cache
        self.disk_cache.set(**item.params())


class TensorLRUCache_Keyspace(TensorLRUCache_LockBase):
    def __init__(self, key_prefix, wrapped):
        super().__init__(nullcontext())
        self._key_prefix = key_prefix
        self._wrapped = wrapped

    def _get(self, key: str) -> TensorMetadataExpiresTuple:
        key = self._key_prefix + key
        return self._wrapped._get(key)

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None):
        key = self._key_prefix + key
        return self._wrapped._set(
            key, tensors, metadata=metadata, expires=expires, max_age=max_age
        )

    def __contains__(self, key):
        key = self._key_prefix + key
        return key in self._wrapped

    def keyspace(self, key_prefix):
        return self._wrapped.keyspace(key_prefix + self._key_prefix)
