import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from safetensors import torch as safe_torch

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

NOT_PASSED_MARKER = object()


@dataclass
class CacheDetails:
    key: str
    access_ctr: int
    size: int
    expires: float | None = None
    tensors: dict | None = None
    metadata: Any | None = None
    path: str | None = None

    def params(self):
        return {
            k: v for k, v in asdict(self).items() if k not in {"access_ctr", "path"}
        }


class CacheKeyError(KeyError):
    pass


class CacheExpiredError(ValueError):
    pass


class TensorLRUCache_LockBase:
    def __init__(self, lock=None):
        self.lock = lock or threading.Lock()

    def get(self, key):
        with self.lock:
            return self._get(key)

    def metadata(self, key):
        with self.lock:
            return self._metadata(key)

    def set(self, *args, **kwargs):
        with self.lock:
            return self._set(*args, **kwargs)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, tensors):
        self.set(key, tensors)


class TensorLRUCache_Base(TensorLRUCache_LockBase):
    def __init__(self, limit, lock=None):
        super().__init__(lock)
        self.limit = limit
        self.cache = {}
        self.access_ctr = 0

    @property
    def next_ctr(self):
        self.access_ctr += 1
        return self.access_ctr

    def keys(self):
        return self.cache.keys()

    def _metadata(self, key):
        if key in self.cache:
            return self.cache[key].metadata

        raise CacheKeyError(key)

    def __contains__(self, key):
        return key in self.cache

    def _size_of(self, tensors):
        return sum((t.nelement() * t.element_size() for t in tensors.values()))

    def _needs_evicting(self):
        items = self.cache.values()
        items = sorted(items, key=lambda entry: entry.access_ctr)

        total = 0
        while items and total < self.limit:
            total += items.pop().size

        if items:
            print(
                f"{total} bytes in cache, {len(items)} too many items in mem cache, evicting"
            )
        else:
            print(f"{total} bytes in cache")

        yield from items


class TensorLRUCache_Mem(TensorLRUCache_Base):
    def __init__(self, limit, lock=None, on_evict=None):
        super().__init__(limit=limit, lock=lock)
        self.on_evict = on_evict

    def _get(self, key):
        if key in self.cache:
            self.cache[key].access_ctr = self.next_ctr
            return self.cache[key].tensors

        raise CacheKeyError(key)

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None, size=None):
        if size is None:
            size = self._size_of(tensors)
        if expires is None:
            expires = time.time() + max_age if max_age else None

        self.cache[key] = CacheDetails(
            key=key,
            access_ctr=self.next_ctr,
            size=size,
            tensors=tensors,
            metadata=metadata,
            expires=expires,
        )

        self.__evict()

    def __evict(self):
        for item in self._needs_evicting():
            if self.on_evict is not None:
                print(f"Evicting {item.key}")
                self.on_evict(item)

            del self.cache[item.key]


class TensorLRUCache_Disk(TensorLRUCache_Base):
    def __init__(self, limit, basepath=None, lock=None):
        super().__init__(limit=limit, lock=lock)
        self.tempdir = tempfile.TemporaryDirectory(dir=basepath)
        self.basepath = self.tempdir.name
        print(self.basepath)

    def _get(self, key):
        if key in self.cache:
            self.cache[key].access_ctr = self.next_ctr
            return safe_torch.load_file(self.cache[key].path)

        raise CacheKeyError()

    def _set(self, key, tensors, metadata=None, expires=None, max_age=None, size=None):
        if size is None:
            size = self._size_of(tensors)
        if expires is None:
            expires = time.time() + max_age if max_age else None

        ctr = self.next_ctr

        if key in self.cache:
            path = self.cache[key].path
        else:
            path = os.path.join(self.basepath, f"cache_{ctr}.safetensors")

        safe_torch.save_file(tensors, path)

        self.cache[key] = CacheDetails(
            key=key,
            access_ctr=ctr,
            size=size,
            path=path,
            metadata=metadata,
            expires=expires,
        )

        self.__evict()

    def __evict(self):
        for item in self._needs_evicting():
            os.remove(item.path)
            del self.cache[item.key]


class TensorLRUCache_Dual(TensorLRUCache_LockBase):
    def __init__(self, basepath=None, memlimit=1 * GB, disklimit=10 * GB):
        super().__init__(threading.RLock())
        self.disk_cache = TensorLRUCache_Disk(
            disklimit, basepath=basepath, lock=self.lock
        )
        self.mem_cache = TensorLRUCache_Mem(memlimit, lock=self.lock)

        self.mem_cache.on_evict = self._on_mem_evict

    def _get(self, key, default=NOT_PASSED_MARKER):
        try:
            return self.mem_cache.get(key)
        except CacheKeyError:
            pass

        try:
            res = self.disk_cache.get(key)
            self.mem_cache.set(key, res)
            return res
        except CacheKeyError:
            pass

        if default is not NOT_PASSED_MARKER:
            return default

        raise CacheKeyError(key)

    def _metadata(self, key):
        try:
            return self.mem_cache.metadata(key)
        except CacheKeyError:
            return self.disk_cache.metadata(key)

    def _set(self, *args, **kwargs):
        self.mem_cache.set(*args, **kwargs)

    def keys(self):
        return set(self.mem_cache.keys()) | set(self.disk_cache.keys())

    def __contains__(self, key):
        return key in self.mem_cache or key in self.disk_cache

    def _on_mem_evict(self, item):
        self.disk_cache.set(**item.params())
