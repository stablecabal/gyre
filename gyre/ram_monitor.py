from dataclasses import dataclass
from pathlib import Path
import threading
import time
import ctypes
import uuid

import torch
import psutil
import pynvml


def mb(v):
    return f"{v / 1024 / 1024 :.2f}MB"


UPDATE_PERIOD = 0.001


class CudaMapper:
    def __init__(self):
        self.mapping: dict[int, uuid.UUID] | None = None

    def _load_first_of(self, candidates):
        for libname in candidates:
            try:
                return ctypes.CDLL(str(libname))
            except OSError:
                continue

    def _build_mapping_cuda(self):
        self.mapping = {}

        libnames = ("libcuda.so", "libcuda.so.1", "libcuda.dylib", "cuda.dll")

        cuda = self._load_first_of(libnames)
        if not cuda:
            raise OSError("could not load any of: " + " ".join(libnames))

        CUDA_SUCCESS = 0
        device = ctypes.c_int()
        cu_uuid = b" " * 16

        if cuda.cuInit(0) != CUDA_SUCCESS:
            raise RuntimeError("Couldn't initialise CUDA for RAM monitor")

        for i in range(torch.cuda.device_count()):
            if cuda.cuDeviceGet(ctypes.byref(device), i) != CUDA_SUCCESS:
                raise RuntimeError(f"Couldn't get device handle for CUDA index {i}")

            if (
                cuda.cuDeviceGetUuid_v2(ctypes.c_char_p(cu_uuid), device)
                != CUDA_SUCCESS
            ):
                raise RuntimeError(f"Couldn't get UUID for CUDA index {i}")

            self.mapping[i] = uuid.UUID(bytes=cu_uuid)

    def _build_mapping_cudart(self):
        self.mapping = {}

        pytorch_libdir = Path(torch.__file__).parent / "lib"

        cudart = self._load_first_of(pytorch_libdir.glob("*cudart*"))
        if not cudart:
            raise OSError("could not load any cudart version")

        CUDA_SUCCESS = 0
        buffer = b"  " * 1024

        for i in range(torch.cuda.device_count()):
            if (
                cudart.cudaGetDeviceProperties(ctypes.c_char_p(buffer), i)
                != CUDA_SUCCESS
            ):
                raise RuntimeError(f"Couldn't get UUID for CUDA index {i}")

            self.mapping[i] = uuid.UUID(bytes=buffer[256 : 256 + 16])

    def get_uuid(self, idx):
        if self.mapping is None:
            try:
                self._build_mapping_cuda()
            except Exception as e:
                print("Couldn't map device UUIDs via cuda, falling back to cudart")
                self._build_mapping_cudart()

        assert self.mapping is not None
        return self.mapping.get(idx)

    def get_handle(self, idx):
        uuid = str(self.get_uuid(idx)).encode("ascii")

        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            if pynvml.nvmlDeviceGetUUID(h).endswith(uuid):
                return h


@dataclass
class CudaDevice:
    handle: pynvml.c_nvmlUnit_t | None = None
    vram_current = 0
    vram_max_usage = 0
    vram_total = 0


class RamMonitor(threading.Thread):
    stop_flag = False
    ram_current = 0
    ram_max_usage = 0
    ram_total = 0
    cuda_devices = {}

    total = -1

    def __init__(self):
        threading.Thread.__init__(self)

        try:
            pynvml.nvmlInit()
        except Exception:
            print("Unable to initialize NVIDIA management. No VRAM stats. \n")
        else:
            cuda_mapper = CudaMapper()
            self.cuda_devices = {
                idx: CudaDevice(handle=cuda_mapper.get_handle(idx))
                for idx in range(torch.cuda.device_count())
            }

    def as_str(self, ram_attr, vram_attr):
        ram_val = getattr(self, ram_attr)
        vram_val = {
            i: getattr(device, vram_attr) for i, device in self.cuda_devices.items()
        }

        parts = [f"RAM={mb(ram_val)}"]

        if vram_val:
            parts.extend((f"CUDA:{i}={mb(v)}" for i, v in vram_val.items()))

        return ", ".join(parts)

    def run(self):
        ps = psutil.Process()

        self.loop_lock = threading.Lock()

        print("Recording max memory usage...")
        self.ram_total = psutil.virtual_memory().total

        for device in self.cuda_devices.values():
            device.vram_total = pynvml.nvmlDeviceGetMemoryInfo(device.handle).total

        total_str = self.as_str("ram_total", "vram_total")
        print(f"Total available RAM: {total_str}")

        while not self.stop_flag:
            self.ram_current = ps.memory_info().rss
            self.ram_max_usage = max(self.ram_max_usage, self.ram_current)

            for device in self.cuda_devices.values():
                device.vram_current = pynvml.nvmlDeviceGetMemoryInfo(device.handle).used
                device.vram_max_usage = max(device.vram_max_usage, device.vram_current)

            if self.loop_lock.locked():
                self.loop_lock.release()

            time.sleep(UPDATE_PERIOD)

        print("Stopped recording.")
        pynvml.nvmlShutdown()

    def print(self):
        # Wait for the update loop to run at least once
        self.loop_lock.acquire(timeout=0.5)

        current_str = self.as_str("ram_current", "vram_current")
        peak_str = self.as_str("ram_max_usage", "vram_max_usage")
        print(f"Current RAM: {current_str} | Peak RAM: {peak_str}")

    def read(self):
        result = dict(ram_max=self.ram_max_usage, ram_total=self.ram_total)

        for i, device in self.cuda_devices.items():
            result[f"cuda:{i}_max"] = device.vram_max_usage
            result[f"cuda:{i}_total"] = device.vram_total

    def read_and_reset(self):
        result = self.read()

        self.ram_current = self.ram_max_usage = 0
        for device in self.cuda_devices.values():
            device.vram_current = device.vram_max_usage = 0

        return result

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop()
        return self.read()
