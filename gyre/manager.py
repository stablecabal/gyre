import glob
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from fnmatch import fnmatch
from types import SimpleNamespace as SN
from typing import Any, Callable, Iterable, Literal
from urllib.parse import urlparse

import generation_pb2
import huggingface_hub
import pynvml
import safetensors.torch
import torch
import yaml
from diffusers import ModelMixin, pipelines
from diffusers.configuration_utils import FrozenDict
from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline,
    is_safetensors_compatible,
)
from huggingface_hub.file_download import http_get
from transformers import PreTrainedModel

from gyre import civitai, ckpt_utils, torch_safe_unpickler
from gyre.constants import IS_DEV, sd_cache_home
from gyre.hints import HintsetManager
from gyre.monitoring_queue import MonitoringQueue
from gyre.pipeline import pipeline_meta
from gyre.pipeline.model_utils import clone_model
from gyre.pipeline.pipeline_wrapper import DiffusionPipelineWrapper, PipelineWrapper
from gyre.pipeline.upscalers.diffusers_upscaler_wrapper import (
    DiffusionUpscalerPipelineWrapper,
)

logger = logging.getLogger(__name__)

DEFAULT_LIBRARIES = {
    "StableDiffusionPipeline": "stable_diffusion",
    "StableDiffusionUpscalePipeline": "stable_diffusion",
    "StableDiffusionLatentUpscalePipeline": "stable_diffusion",
    "UnifiedPipeline": "gyre.pipeline.unified_pipeline",
    "DiffusersDepthPipeline": "gyre.pipeline.depth.diffusers_depth_pipeline",
    "MidasModelWrapper": "gyre.pipeline.depth.midas_model_wrapper",
    "MidasDepthPipeline": "gyre.pipeline.depth.midas_depth_pipeline",
    "ZoeModelWrapper": "gyre.pipeline.depth.zoe_model_wrapper",
    "ZoeDepthPipeline": "gyre.pipeline.depth.zoe_depth_pipeline",
    "T2iAdapter": "gyre.pipeline.t2i_adapter",
    "ControlNetModel": "gyre.pipeline.controlnet",
    "HED": "gyre.pipeline.hinters.models.hed",
    "HedPipeline": "gyre.pipeline.hinters.hed_pipeline",
    "DexiNed": "kornia.filters",
    "DexinedPipeline": "gyre.pipeline.hinters.dexined_pipeline",
    "MmLoader": "gyre.pipeline.hinters.mm_loader",
    "MmsegPipeline": "gyre.pipeline.hinters.mmseg_pipeline",
    "MmposePipeline": "gyre.pipeline.hinters.mmpose_pipeline",
    "InSPyReNet_SwinB": "gyre.pipeline.hinters.inspyrenet.InSPyReNet",
    "InSPyReNetPipeline": "gyre.pipeline.hinters.inspyrenet_pipeline",
    "BaenormalLoader": "gyre.pipeline.hinters.baenormal_loader",
    "BaenormalPipeline": "gyre.pipeline.hinters.baenormal_pipeline",
    "DrawingGenerator": "gyre.pipeline.hinters.models.informative_drawings",
    "InformativeDrawingPipeline": "gyre.pipeline.hinters.informative_drawing_pipeline",
    "UpscalerLoader": "gyre.pipeline.upscalers.upscaler_loader",
    "UpscalerPipeline": "gyre.pipeline.upscalers.upscaler_pipeline",
}


TYPE_CLASSES = {
    "vae": "diffusers.AutoencoderKL",
    "unet": "diffusers.UNet2DConditionModel",
    "inpaint_unet": "diffusers.UNet2DConditionModel",
    "clip_model": "transformers.CLIPModel",
    "feature_extractor": "transformers.CLIPFeatureExtractor",
    "tokenizer": "transformers.CLIPTokenizer",
    "clip_tokenizer": "transformers.CLIPTokenizer",
    "text_encoder": "transformers.CLIPTextModel",
    "inpaint_text_encoder": "transformers.CLIPTextModel",
    "depth_estimator": "transformers.DPTForDepthEstimation",
    "midas_depth_estimator": "MidasModelWrapper",
    "zoe_depth_estimator": "ZoeModelWrapper",
    "t2i_adapter": "T2iAdapter",
    "controlnet": "ControlNetModel",
}


def clip(val, minval, maxval):
    return max(min(val, maxval), minval)


class EngineMode(object):
    def __init__(
        self,
        vram_optimisation_level=0,
        vram_overrides={},
        enable_cuda=True,
        enable_mps=False,
        vram_fraction=1.0,
    ):
        self._vramO = vram_optimisation_level
        self._overrides = vram_overrides
        self._enable_cuda = enable_cuda
        self._enable_mps = enable_mps
        self._vram_fraction = vram_fraction

    @property
    def device(self):
        self._hasCuda = (
            self._enable_cuda
            and getattr(torch, "cuda", False)
            and torch.cuda.is_available()
        )
        self._hasMps = (
            self._enable_mps
            and getattr(torch.backends, "mps", False)
            and torch.backends.mps.is_available()
        )
        return "cuda" if self._hasCuda else "mps" if self._hasMps else "cpu"

    @property
    def attention_slice(self):
        return self._overrides.get(
            "attention_slice",
            self.device == "cuda" and self._vramO > 0,
        )

    @property
    def tile_vae(self):
        return self._overrides.get(
            "tile_vae",
            False,
        )

    @property
    def fp16(self):
        return self._overrides.get(
            "fp16",
            self.device == "cuda" and self._vramO > 1,
        )

    @property
    def cfg_execution(self) -> Literal["parallel", "sequential"]:
        return self._overrides.get(
            "cfg_execution", "sequential" if self._vramO > 4 else "parallel"
        )

    @property
    def gpu_offload(self):
        return self._overrides.get(
            "gpu_offload",
            self.device == "cuda" and self._vramO > 2,
        )

    @property
    def model_vram_limit(self):
        if not self.gpu_offload:
            return -1

        if "model_vram_limit" in self._overrides:
            return self._overrides["model_vram_limit"]

        GB = 1024 * 1024 * 1024

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            vram_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        except:
            vram_total = 4 * GB

        vram_total = vram_total * self._vram_fraction

        if self._vramO <= 3:
            return clip(vram_total * 0.5, 2 * GB, vram_total - 2 * GB)
        elif self._vramO <= 4:
            return clip(3 * GB, 2 * GB, vram_total - 2 * GB)
        else:
            return clip(2 * GB, 2 * GB, vram_total - 2 * GB)

    @property
    def model_max_limit(self):
        if not self.gpu_offload:
            return -1

        if "model_max_limit" in self._overrides:
            return self._overrides["model_max_limit"]

        return 1 if self._vramO >= 5 else -1


class BatchMode:
    def __init__(self, autodetect=False, points=None, simplemax=1, safety_margin=0.2):
        self.autodetect = autodetect
        self.points = json.loads(points) if isinstance(points, str) else points
        self.simplemax = simplemax
        self.safety_margin = safety_margin

    def batchmax(self, pixels):
        if self.points:
            # If pixels less than first point, return that max
            if pixels <= self.points[0][0]:
                return self.points[0][1]

            # Linear interpolate between bracketing points
            pairs = zip(self.points[:-1], self.points[1:])
            for pair in pairs:
                if pixels >= pair[0][0] and pixels <= pair[1][0]:
                    i = (pixels - pair[0][0]) / (pair[1][0] - pair[0][0])
                    return math.floor(pair[0][1] + i * (pair[1][1] - pair[0][1]))

            # Off top of points - assume max of 1
            return 1

        if self.simplemax is not None:
            return self.simplemax

        return 1

    def run_autodetect(self, manager, resmax=2048, resstep=256):
        torch.cuda.set_per_process_memory_fraction(1 - self.safety_margin)

        params = SN(
            height=512,
            width=512,
            cfg_scale=7.5,
            sampler=generation_pb2.SAMPLER_DDIM,
            eta=0,
            steps=8,
            strength=1,
            seed=-1,
        )

        l = 32  # Starting value - 512x512 fails inside PyTorch at 32, no amount of VRAM can help

        pixels = []
        batchmax = []

        for x in range(512, resmax, resstep):
            params.width = x
            print(f"Determining max batch for {x}")
            # Quick binary search
            r = l  # Start with the max from the previous run
            l = 1

            while l < r - 1:
                b = (l + r) // 2
                print(f"Trying {b}")
                try:
                    with manager.with_engine() as pipe:
                        pipe.generate(["A Crocodile"] * b, params, suppress_output=True)
                except Exception as e:
                    r = b
                else:
                    l = b

            print(f"Max for {x} is {l}")

            pixels.append(params.width * params.height)
            batchmax.append(l)

            if l == 1:
                print(f"Max res is {x}x512")
                break

        self.points = list(zip(pixels, batchmax))
        print(
            "To save these for next time, use these for batch_points:",
            json.dumps(self.points),
        )

        torch.cuda.set_per_process_memory_fraction(1.0)


class ModelSet:
    def __init__(self, data: dict[str, Any] = {}) -> None:
        self.__data = {}
        self.__data.update(data)
        self.__frozen = False

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ModelSet":
        return ModelSet(kwargs)

    def freeze(self):
        self.__frozen = True

    def update(self, other: "dict | ModelSet"):
        if self.__frozen:
            raise ValueError("ModelSet is frozen")

        if isinstance(other, ModelSet):
            self.__data.update(other.__data)
        else:
            self.__data.update(other)

    def copy(self):
        return ModelSet(self.__data)

    def get(self, key, default=None):
        return self.__data.get(key, default)

    def keys(self):
        return self.__data.keys()

    def values(self):
        return self.__data.values()

    def items(self):
        return self.__data.items()

    def first_key(self):
        for key in self.__data.keys():
            return key

    def first(self):
        for value in self.__data.values():
            return value

    def is_empty(self):
        return len(self.__data) == 0

    def is_singular(self):
        return len(self.__data) == 1

    def as_dict(self):
        res = {}
        res.update(self.__data)
        return res

    def __len__(self):
        return len(self.__data)

    def __contains__(self, item):
        return item in self.__data

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise KeyError(key)

        return self.__data[key]

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)

        elif self.__frozen:
            raise ValueError("ModelSet is frozen")

        else:
            self.__data[key] = value

    def __getitem__(self, key):
        return self.__data[key]

    def __setitem__(self, key, value):
        if self.__frozen:
            raise ValueError("ModelSet is frozen")

        self.__data[key] = value


class EngineSpec:
    @classmethod
    def is_engine_spec(cls, data: dict):
        return "id" in data or "model_id" in data

    def __init__(self, data: dict | None = None):
        if data is None:
            data = {}

        self._data = {k.lower(): v for k, v in data.items()}

    @property
    def human_id(self) -> str:
        if self.id:
            return f"Engine {self.id}"
        else:
            return f"Model {self.model_id}"

    @property
    def is_engine(self) -> bool:
        return "id" in self._data

    @property
    def is_model(self) -> bool:
        return "model_id" in self._data

    @property
    def enabled(self) -> bool:
        return self._data.get("enabled", True)

    @property
    def visible(self) -> bool:
        return self.enabled and self._data.get("visible", True)

    @property
    def type(self) -> str:
        return self._data.get("type", "pipeline").lower()

    @property
    def task(self) -> str | None:
        if self.type == "pipeline":
            return self._data.get("task", "generate").lower()
        else:
            return None

    @property
    def class_name(self) -> str | None:
        default = None

        if self.type == "pipeline":
            if self.task == "depth":
                default = "MidasDepthPipeline"
            else:
                default = "UnifiedPipeline"

        return self._data.get("class", default)

    @property
    def fp16(self) -> Literal["auto", "only", "local", "never", "prevent"]:
        res = self._data.get("fp16", "auto").lower()
        values = {"auto", "only", "local", "never", "prevent"}
        assert res in values, f"Invalid fp16 value {res}"
        return res

    @property
    def model_is_empty(self) -> bool:
        return self.model and self.model == "@empty"

    @property
    def model_is_reference(self) -> bool:
        return self.model and self.model[0] == "@" and not self.model_is_empty

    @property
    def local_model(self) -> str | None:
        path = self._data.get("local_model")
        if not path:
            path = self.model_id
        return path

    @property
    def local_model_fp16(self) -> str | None:
        path = self._data.get("local_model_fp16")
        if not path:
            path = self.local_model
            if path:
                path += "-fp16"
        return path

    def get(self, __name: str, *args) -> Any:
        return getattr(self, __name, *args)

    def __getattr__(self, __name: str) -> Any:
        # Special case, if the attribute is "class", rename to class_name
        if __name == "class":
            return self.class_name

        return self._data.get(__name)

    def __contains__(self, __name) -> bool:
        return __name in self._data


class HintsetSpec:
    @classmethod
    def is_hintset_spec(cls, data: dict):
        return "hintset_id" in data

    def __init__(self, data: dict | None = None):
        if data is None:
            data = {}

        self._data = {k.lower(): v for k, v in data.items()}

    def items(self):
        for k, v in self._data.items():
            if k != "hintset_id":
                yield k, v

    def get(self, __name: str, *args) -> Any:
        return getattr(self, __name, *args)

    def __getattr__(self, __name: str) -> Any:
        return self._data.get(__name)

    def __contains__(self, __name) -> bool:
        return __name in self._data


@dataclass
class DeviceQueueSlot:
    device: torch.device
    pipeline: DiffusionPipeline | None = None


class EngineNotFoundError(Exception):
    pass


class EngineNotReadyError(Exception):
    pass


def all_same(items):
    return all(x == items[0] for x in items)


@dataclass
class RepoFile:
    full: str
    name: str
    dtype: str | None
    kind: str

    @classmethod
    def from_str(cls, f, **overrides):
        *parts, dtype, kind = f.split(".")
        if not parts:
            parts, dtype = [dtype], None
        kwargs = dict(name=".".join(parts), dtype=dtype, kind=kind)
        kwargs.update(overrides)
        return RepoFile(full=f, **kwargs)

    model_kinds = {
        "ckpt",
        "bin",
        "pt",
        "pth",
        "safetensors",
        "msgpack",
        "h5",
    }


_skip = object()


class RepoFileSet:
    def __init__(self, files: list[RepoFile]):
        self.files = files

    def _parse_args(self, name, dtype, kind):
        if name is not _skip and not isinstance(name, set):
            name = {name}
        if dtype is not _skip and not isinstance(dtype, set):
            dtype = {dtype}
        if kind is not _skip and not isinstance(kind, set):
            kind = {kind}

        return name, dtype, kind

    def _test(self, f, name, dtype, kind):
        return (
            (name is _skip or f.name in name)
            and (dtype is _skip or f.dtype in dtype)
            and (kind is _skip or f.kind in kind)
        )

    def _find(self, name, dtype, kind, exclusive=False):
        return [f for f in self.files if self._test(f, name, dtype, kind) != exclusive]

    def find(self, attr="name", name=_skip, dtype=_skip, kind=_skip):
        name, dtype, kind = self._parse_args(name, dtype, kind)
        return {getattr(f, attr) for f in self._find(name, dtype, kind)}

    def remove(self, name=_skip, dtype=_skip, kind=_skip):
        name, dtype, kind = self._parse_args(name, dtype, kind)
        self.files = self._find(name, dtype, kind, exclusive=True)

    def without(self, name=_skip, dtype=_skip, kind=_skip):
        name, dtype, kind = self._parse_args(name, dtype, kind)
        return RepoFileSet(self._find(name, dtype, kind, exclusive=True))

    @staticmethod
    def safetensor_equivalents(names):
        res = set()
        for name in names:
            head, tail = os.path.split(name)
            if tail == "pytorch_model":
                tail = "model"

            res.add(os.path.join(head, tail))
        return res


class EngineManager(object):
    def __init__(
        self,
        engines,
        weight_root="./weights",
        refresh_models=None,
        refresh_on_error=False,
        mode=EngineMode(),
        nsfw_behaviour="block",
        batchMode=BatchMode(),
        ram_monitor=None,
    ):
        self.engines = [
            EngineSpec(spec) for spec in engines if EngineSpec.is_engine_spec(spec)
        ]
        self.hintsets = [
            HintsetSpec(spec) for spec in engines if HintsetSpec.is_hintset_spec(spec)
        ]
        self._defaults = {}

        self.status: Literal["created", "loading", "ready"] = "created"

        # Models that are explictly loaded with a model_id and can be referenced
        self._models: dict[str, ModelSet] = {}
        # Models for each engine
        self._engine_models: dict[str, ModelSet] = {}
        # Hintsets
        self._hintsets: dict[str, HintManager] = {}

        self._activeId = None
        self._active = None

        self._weight_root = weight_root
        self._refresh_models = refresh_models
        self._refresh_on_error = refresh_on_error

        self._mode = mode
        self._batchMode = batchMode
        self._nsfw = nsfw_behaviour
        self._token = os.environ.get("HF_API_TOKEN", True)

        self._ram_monitor = ram_monitor

        # A queue that holds all available slots, so threads can take a slot off the pile
        self._device_queue = MonitoringQueue()
        # All device slots, whether in the queue or in use. For monitoring purposes only.
        self._device_slots = []
        # A set of pipelines created for a specific engine id, which are not currently allocated to a slot
        # We may need more than one copy of a pipeline if it's being run more than once in parallel.
        self._available_pipelines: dict[str, deque] = {}

        for i in range(torch.cuda.device_count()):
            device_queue_slot = DeviceQueueSlot(device=torch.device("cuda", i))
            self._device_queue.put(device_queue_slot)
            self._device_slots.append(device_queue_slot)

    @property
    def mode(self):
        return self._mode

    @property
    def batchMode(self):
        return self._batchMode

    def _get_local_path(self, spec: EngineSpec, fp16=False):
        path = None

        # Pick the right path
        if fp16:
            path = spec.local_model_fp16
        else:
            path = spec.local_model

        # Throw error if no such key in spec
        if not path:
            raise ValueError(f"No local model field was provided")
        # Add path to weight root if not absolute
        if not os.path.isabs(path):
            path = os.path.join(self._weight_root, path)
        # Normalise
        path = os.path.normpath(path)
        # Throw error if result isn't a directory
        if not os.path.isdir(path):
            raise ValueError(f"Path '{path}' isn't a directory")

        return path

    def _get_hf_path(self, spec: EngineSpec, local_only=True):
        extra_kwargs = {}

        model_path = spec.model

        # If no model_path is provided, don't try and download
        if not model_path:
            raise ValueError("No remote model name was provided")

        # Support providing fixed revision
        revision = spec.revision

        if revision:
            extra_kwargs["revision"] = revision

        # Handle various fp16 modes (local, never and prevent are all the same for this method)
        require_fp16 = self.mode.fp16 and spec.fp16 == "only"
        prefer_fp16 = self.mode.fp16 and spec.fp16 == "auto"
        has_fp16 = None

        # In local_only mode it's very hard to determine if local weights are fp16 or not
        # It's not used anyway, so deprecate "only" mode
        if require_fp16:
            logger.warn("fp16: only is deprecated. Falling back to fp16: auto")
            prefer_fp16 = True

        subfolder = f"{spec.subfolder}/" if spec.subfolder else ""

        # Read any specified ignore or allow patterns
        def build_patterns(patterns):
            if not patterns:
                return []
            elif isinstance(patterns, str):
                return [patterns]
            else:
                return patterns

        ignore_patterns = build_patterns(spec.ignore_patterns)
        allow_patterns = build_patterns(spec.allow_patterns)

        # Adjust if subfolder is set
        if subfolder:
            ignore_patterns = [f"{subfolder}{pattern}" for pattern in ignore_patterns]
            allow_patterns = [f"{subfolder}{pattern}" for pattern in allow_patterns]
            if not allow_patterns:
                allow_patterns = [f"{subfolder}*"]

        use_auth_token = self._token if spec.use_auth_token else False

        if use_auth_token:
            extra_kwargs["use_auth_token"] = use_auth_token

        try:
            # If we're not loading from local_only, do some extra logic to avoid downloading
            # other unusused large files in the repo unnessecarily (like .ckpt files and
            # the .safetensors version of .ckpt files )
            if not local_only:
                # Get a list of files, split into path and extension
                repo_info = None
                overrides = {}
                if (not revision) and prefer_fp16:
                    try:
                        repo_info = huggingface_hub.model_info(
                            model_path, revision="fp16", **extra_kwargs
                        )
                        overrides["dtype"] = "fp16"
                    except huggingface_hub.utils.RevisionNotFoundError as e:
                        pass

                if repo_info is None:
                    repo_info = huggingface_hub.model_info(model_path, **extra_kwargs)

                # Read out the list of files, filtering by any ignore / allow
                repo_files = list(
                    huggingface_hub.utils.filter_repo_objects(
                        [f.rfilename for f in repo_info.siblings],
                        ignore_patterns=ignore_patterns if ignore_patterns else None,
                        allow_patterns=allow_patterns if allow_patterns else None,
                    )
                )

                # Convert strings into RepoFile objects
                repo_file_details = [
                    RepoFile.from_str(f, **overrides) for f in repo_files if "." in f
                ]

                # And then collect the ones that look like models in a set
                model_files = RepoFileSet(
                    [f for f in repo_file_details if f.kind in RepoFile.model_kinds]
                )

                # We need to figure out what files to download. Make these assumptions:
                # - Any .safetensors that match some other .model (ignoring dtype) are the same underlying type
                # - If we don't have a clear type (diffusers or ckpt), we have to include all safetensors

                # What kinds of model exist?
                has = {k: k in model_files.find("kind") for k in RepoFile.model_kinds}

                # If we have bin, pt or pth files, remove any safetensors that match ckpts
                # to make future logic easier

                if has["bin"] or has["pt"] or has["pth"]:
                    test_files = model_files.without(name=model_files.find(kind="ckpt"))
                else:
                    test_files = model_files

                # Pick the kind and specific (bare) names
                kind = names = None

                # 1st choice: bin (or even better, safetensors that match bin)
                if has["bin"]:
                    names = test_files.find(kind="bin")
                    equivalents = RepoFileSet.safetensor_equivalents(names)
                    safetensors = test_files.find(kind="safetensors")

                    if equivalents - safetensors:
                        logger.debug(
                            f"Diffusers models were missing some safetensors ({equivalents - safetensors})"
                        )
                        kind = "bin"
                    else:
                        names = equivalents
                        kind = "safetensors"
                # 2nd choice: safetensors
                elif has["safetensors"]:
                    names = test_files.find(kind="safetensors")
                    kind = "safetensors"
                # 3rd choice, ".pt" or ".pth"
                elif has["pt"]:
                    names = test_files.find(kind="pt")
                    kind = "pt"
                elif has["pth"]:
                    names = test_files.find(kind="pth")
                    kind = "pth"
                # 4th choice: ckpt (or safetensors that match if possible)
                elif has["ckpt"]:
                    names = test_files.find(kind="ckpt")
                    safetensors = test_files.find(kind="safetensors")

                    if names - safetensors:
                        logger.debug(
                            f"Checkpoints were some safetensors ({names - safetensors})"
                        )
                        kind = "ckpt"
                    else:
                        kind = "safetensors"

                else:
                    raise EnvironmentError(
                        "Repo {model_path} doesn't appear to contain any model files."
                    )

                if spec.safe_only and kind != "safetensors":
                    raise RuntimeError(
                        "spec.safe_only set, but couldn't find appropriate safetensors files"
                    )

                if prefer_fp16:
                    if names - model_files.find(name=names, dtype="fp16", kind=kind):
                        has_fp16 = prefer_fp16 = False
                    else:
                        has_fp16 = True

                logger.debug(
                    f"Model chosen {kind}{', fp16' if has_fp16 else ''}, names: {names}"
                )

                chosen = [
                    model_files.find(
                        "full", name=name, dtype="fp16" if has_fp16 else None, kind=kind
                    ).pop()
                    for name in names
                ]

                ignore_patterns += [
                    f for f in model_files.find("full") if f not in chosen
                ]

                if ignore_patterns:
                    extra_kwargs["ignore_patterns"] = ignore_patterns
                if allow_patterns:
                    extra_kwargs["allow_patterns"] = allow_patterns

            if (not revision) and prefer_fp16:
                try:
                    base = huggingface_hub.snapshot_download(
                        model_path,
                        repo_type="model",
                        local_files_only=local_only,
                        revision="fp16",
                        **extra_kwargs,
                    )
                    return os.path.join(base, subfolder) if subfolder else base
                except (
                    FileNotFoundError,
                    huggingface_hub.utils.RevisionNotFoundError,
                ):
                    pass

            base = huggingface_hub.snapshot_download(
                model_path,
                repo_type="model",
                local_files_only=local_only,
                **extra_kwargs,
            )

            return os.path.join(base, subfolder) if subfolder else base

        except Exception as e:
            if local_only:
                raise ValueError("Couldn't query local HuggingFace cache." + str(e))
            else:
                raise ValueError("Downloading from HuggingFace failed." + str(e))

    def _get_hf_forced_path(self, spec: EngineSpec):
        model_path = spec.model

        # If no model_path is provided, don't try and download
        if not model_path:
            raise ValueError("No remote model name was provided")

        try:
            repo_info = next(
                (
                    repo
                    for repo in huggingface_hub.scan_cache_dir().repos
                    if repo.repo_id == model_path
                )
            )
            hashes = [revision.commit_hash for revision in repo_info.revisions]
            huggingface_hub.scan_cache_dir().delete_revisions(*hashes).execute()
        except:
            pass

        return self._get_hf_path(spec, local_only=False)

    def _get_civitai_path(self, spec: EngineSpec, local_only=True):
        ref = civitai.parse_url(spec.model)
        return civitai.get_model(ref, local_only=local_only)

    def _get_url_path(self, spec: EngineSpec, local_only=True):
        urls = spec.model or spec.urls

        if not urls:
            raise ValueError("No URL was provided")

        if isinstance(urls, str):
            id = hashlib.sha1(urls.encode("utf-8")).hexdigest()
            _, filename = os.path.split(urlparse(urls).path)
            urls = {filename: urls}
        else:
            id = urls["id"]
            urls = {k: v for k, v in urls.items() if k != "id"}

        cache_path = os.path.join(sd_cache_home, id)
        temp_path = os.path.join(sd_cache_home, "temp")

        if os.path.isdir(cache_path):
            exists = {
                name: os.path.isfile(os.path.join(cache_path, name))
                for name in urls.keys()
            }
            if all(exists.values()):
                return cache_path
            elif local_only:
                raise ValueError(
                    f"Items missing from cache: {[name for name, exist in exists.items() if not exist]}"
                )
        elif local_only:
            raise ValueError("No local cache for URL")

        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)

        for name, url in urls.items():
            full_name = os.path.join(cache_path, name)
            if os.path.exists(full_name):
                continue

            temp_name = None
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=temp_path, delete=False
            ) as temp_file:
                http_get(url, temp_file)
                temp_name = temp_file.name

            if temp_name:
                os.replace(temp_name, full_name)

        return cache_path

    def _get_weight_path_candidates(self, spec: EngineSpec):
        candidates = []

        def add_candidate(callable, *args, **kwargs):
            candidates.append((callable, args, kwargs))

        model_path = spec.model
        matches_refresh = (
            self._refresh_models
            and model_path
            and any(
                (
                    True
                    for pattern in self._refresh_models
                    if fnmatch(model_path, pattern)
                )
            )
        )

        if not model_path:
            model_source = None
        elif model_path.startswith("https://civitai.com"):
            model_source = self._get_civitai_path
        elif model_path.startswith("https://"):
            model_source = self._get_url_path
        else:
            model_source = self._get_hf_path

        # 1st: If this model should explicitly be refreshed, try refreshing from URL...
        if model_source and matches_refresh:
            add_candidate(model_source, local_only=False)
        # 2nd: If we're in fp16 mode, try loading the fp16-specific local model
        if self.mode.fp16 and spec.fp16 not in {"never", "prevent"}:
            add_candidate(self._get_local_path, fp16=True)
        # 3rd: Try loading the general local model
        if not (self.mode.fp16 and spec.fp16 == "only"):
            add_candidate(self._get_local_path, fp16=False)
        # 4th: Try loading from the existing cache
        if model_source:
            add_candidate(model_source, local_only=True)
        # 5th: If this model wasn't explicitly flagged to be refreshed, try anyway
        if model_source and not matches_refresh:
            add_candidate(model_source, local_only=False)
        # 6th: If configured so, try a forced empty-cache-and-reload from HuggingFace
        if self._refresh_on_error:
            add_candidate(self._get_hf_forced_path)

        return candidates

    def _import_class(self, fqclass_name: str | tuple[str, str]):
        # You can pass in either a (dot seperated) string or a tuple of library, class
        if isinstance(fqclass_name, str):
            *library_name, class_name = fqclass_name.split(".")
            library_name = ".".join(library_name)
        else:
            library_name, class_name = fqclass_name

        if not library_name:
            library_name = DEFAULT_LIBRARIES.get(class_name, None)

        if not library_name:
            raise EnvironmentError(
                f"Don't know the library name for class {class_name}"
            )

        # Is `library_name` a submodule of diffusers.pipelines?
        is_pipeline_module = hasattr(pipelines, library_name)

        if is_pipeline_module:
            # If so, look it up from there
            pipeline_module = getattr(pipelines, library_name)
            class_obj = getattr(pipeline_module, class_name)
        else:
            # else we just import it from the library.
            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name, None)

            # Backwards compatibility - if config asks for transformers.CLIPImageProcessor
            # and we don't have it, use transformers.CLIPFeatureExtractor, that's the old name
            if not class_obj:
                if (
                    library_name == "transformers"
                    and class_name == "CLIPImageProcessor"
                ):
                    class_obj = getattr(library, "CLIPFeatureExtractor", None)

            if not class_obj:
                raise EnvironmentError(
                    f"Config attempts to import {library}.{class_name} that doesn't appear to exist"
                )

        return class_obj

    def _load_module_fallback(
        self,
        path,
        class_obj,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        paths = []
        for pattern in ["*.safetensors", "*.pt", "*.pth"]:
            paths += glob.glob(pattern, root_dir=path)

        paths = list(
            huggingface_hub.utils.filter_repo_objects(
                paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if not paths:
            raise RuntimeError(f"No model found for {class_obj.__name__} at {path}")
        elif len(paths) > 1:
            raise RuntimeError(
                f"Too many posible models found for {class_obj.__name__} at {path}. "
                "Try adding an allow_patterns or ignore_patterns to your model spec."
            )

        model_path = os.path.join(path, paths[0])

        adapter = class_obj(**config)
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path)
        else:
            state_dict = torch.load(model_path, pickle_module=torch_safe_unpickler)

        adapter.load_state_dict(state_dict)

        if torch_dtype != "auto":
            adapter.to(torch_dtype)

        adapter.eval()
        return adapter

    def _parse_class_details(self, fqclass_name):
        factory_name = None
        args = {}

        if fqclass_name:
            if isinstance(fqclass_name, str):
                base_name = fqclass_name
            else:
                base_name = fqclass_name[1]

            # Extract out any arguments in the class name
            if base_name.endswith(")"):
                base_name, argstr = base_name.split("(", 1)
                argstr = argstr[:-1]

                args = yaml.load(
                    "{" + argstr.replace("=", ": ") + "}", Loader=yaml.SafeLoader
                )
                args = {k: None if v == "None" else v for k, v in args.items()}

            # Extract out any loader method override in the class name (must be classmethod)
            if "/" in base_name:
                base_name, factory_name = base_name.split("/", 1)

            if isinstance(fqclass_name, str):
                fqclass_name = base_name
            else:
                fqclass_name = (fqclass_name[0], base_name)

        return fqclass_name, factory_name, args

    def _load_model_from_weights(
        self,
        weight_path: str,
        name: str,
        fqclass_name: str | tuple[str, str] | None = None,
        local_only: bool = False,
        fp16: bool | None = None,
        ignore_patterns=None,
        allow_patterns=None,
    ):
        assert "/" not in name

        def subclass_check(obj, classtype):
            if type(obj) is not type:
                return False
            return issubclass(obj, classtype)

        fqclass_name, factory_name, args = self._parse_class_details(fqclass_name)
        load_method_names = (
            [factory_name] if factory_name else ["from_pretrained", "from_config"]
        )

        if fqclass_name is None:
            fqclass_name = TYPE_CLASSES.get(name, None)

        if fp16 is None:
            fp16 = self.mode.fp16

        if fqclass_name is None:
            raise EnvironmentError(
                f"Type {name} does not specify a class, and there is no default set for it."
            )

        class_obj = self._import_class(fqclass_name)

        load_candidates = [getattr(class_obj, name, None) for name in load_method_names]
        load_candidates = [m for m in load_candidates if m is not None]

        load_method = None
        loading_kwargs = args or {}

        if load_candidates:
            load_method = load_candidates[0]
        else:
            load_method = self._load_module_fallback
            loading_kwargs["class_obj"] = class_obj

        if not load_method:
            raise RuntimeError(f"No load method found for model {class_obj.__name__}")

        init_params = inspect.signature(load_method).parameters

        if fp16 and (
            subclass_check(class_obj, torch.nn.Module) or "torch_dtype" in init_params
        ):
            loading_kwargs["torch_dtype"] = torch.float16

        is_diffusers_model = subclass_check(class_obj, ModelMixin)
        is_transformers_model = subclass_check(class_obj, PreTrainedModel)

        accepts_low_cpu_mem_usage = (
            is_diffusers_model
            or is_transformers_model
            or "low_cpu_mem_usage" in init_params
        )
        accepts_variant = is_diffusers_model or "variant" in init_params

        if accepts_low_cpu_mem_usage:
            loading_kwargs["low_cpu_mem_usage"] = True
        if "ignore_patterns" in init_params:
            loading_kwargs["ignore_patterns"] = ignore_patterns
        if "allow_patterns" in init_params:
            loading_kwargs["allow_patterns"] = allow_patterns

        # check if the module is in a subdirectory
        sub_path = os.path.join(weight_path, name)
        if os.path.isdir(sub_path):
            weight_path = sub_path

        # We can't _know_ if the fp16 variant is loadable without checking the
        # full model details online. So just _try_ and fall back to trying without variant

        # This is _SO DUMB_ - some models call set_default_dtype, but don't properly
        # wrap that in a try / finally to restore the original dtype
        default_dtype = torch.get_default_dtype()

        model = variant_exception = None
        if accepts_variant and fp16:
            try:
                model = load_method(weight_path, variant="fp16", **loading_kwargs)
            except Exception as e:
                variant_exception = e
            finally:
                torch.set_default_dtype(default_dtype)

        if model is None:
            try:
                model = load_method(weight_path, **loading_kwargs)
            except Exception as e:
                if variant_exception is not None:
                    raise e from variant_exception
                else:
                    raise e
            finally:
                torch.set_default_dtype(default_dtype)

        model._source = weight_path
        return model

    def _load_modelset_from_weights(
        self, weight_path, whitelist=None, blacklist=None, **kwargs
    ):
        config_dict = DiffusionPipeline.load_config(weight_path, local_files_only=True)

        if isinstance(whitelist, str):
            whitelist = [whitelist]
        if whitelist:
            whitelist = set(whitelist)
        if isinstance(blacklist, str):
            blacklist = [blacklist]
        if blacklist:
            blacklist = set(blacklist)

        pipeline = {}

        class_items = [
            item for item in config_dict.items() if isinstance(item[1], list)
        ]

        for name, fqclass_name in class_items:
            if whitelist and name not in whitelist:
                continue
            if blacklist and name in blacklist:
                continue
            if fqclass_name[1] is None:
                pipeline[name] = None
                continue

            if name == "safety_checker":
                if self._nsfw == "flag":
                    fqclass_name = "gyre.pipeline.safety_checkers.FlagOnlySafetyChecker"
                elif self._nsfw == "ignore":
                    pipeline[name] = None
                    continue

            pipeline[name] = self._load_model_from_weights(
                weight_path, name, fqclass_name, **kwargs
            )

        return ModelSet(pipeline)

    # mix_* methods copied from https://github.com/huggingface/diffusers/blob/main/examples/community/checkpoint_merger.py

    @staticmethod
    def mix_weighted_sum(alpha, theta0, theta1):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def mix_sigmoid(alpha, theta0, theta1):
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def mix_inv_sigmoid(alpha, theta0, theta1):
        alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
        return theta0 + ((theta1 - theta0) * alpha)

    @staticmethod
    def mix_difference(alpha, theta0, theta1, theta2):
        return theta0 + (theta1 - theta2) * (1.0 - alpha)

    def _mix_models(
        self, mix_method: Callable, models: list[torch.nn.Module], alpha: float
    ) -> torch.nn.Module:
        thetas = [model.state_dict() for model in models]
        result = {}

        for key in thetas[0].keys():
            # Special case for position IDs
            if key == "text_model.embeddings.position_ids":
                base = thetas[0][key]
                result[key] = torch.arange(0, base.shape[1]).unsqueeze(0).to(base)
                continue

            tomix = [theta[key] for theta in thetas]
            shapes = [tensor.shape for tensor in tomix]

            neqidx = [i for i, (u, v) in enumerate(zip(shapes[0], shapes[1])) if u != v]

            # If all the shapes match, easy to mix them
            if all_same(shapes):
                mix = mix_method(alpha, *tomix)

            # Else if the first shape is larger than the others at dim=1, but otherwise equal
            # handle as a special case (mixing into an inpaint unet)
            elif all_same(shapes[1:]) and neqidx == [1]:
                dim1_slice = slice(0, shapes[1][1])
                dim1_mix = mix_method(alpha, tomix[0][:, dim1_slice, :, :], *tomix[1:])

                mix = tomix[0].clone()
                mix[:, dim1_slice, :, :] = dim1_mix

            else:
                raise ValueError(
                    "Can only mix models with the same shapes. "
                    "If you're trying to mix an inpaint unet with another unet, the inpaint unet must come first. "
                    f"Shapes were {shapes}"
                )

            result[key] = mix

        mixed_model = clone_model(models[0], clone_tensors="cpu")
        mixed_model.load_state_dict(result)
        mixed_model._source = "Mix " + ",".join(model._source for model in models)
        return mixed_model

    def _load_mixed_model(self, spec: EngineSpec) -> ModelSet:
        mix = {"alpha": 0.5, "type": "sigmoid"}
        if "mix" in spec:
            mix.update(spec.mix)

        alpha = mix["alpha"]
        if "alpha" in spec:
            alpha = spec.get("alpha")
            print("Deprecation notice: alpha should be part of the mix dictionary")

        mix_type = mix["type"]
        if mix_type not in {"weighted_sum", "sigmoid", "inv_sigmoid", "difference"}:
            raise ValueError(
                "mix.type must be one of weighted_sum, sigmoid, inv_sigmoid, difference"
            )

        mix_method = getattr(self.__class__, "mix_" + mix_type, None)
        if not mix_method:
            raise RuntimeError(f"Couldn't find handler for mix_type {mix_type}")

        # Build the list of models to mix
        models: list[ModelSet] = []

        # Load the primary models. Currently only support 2
        for model in spec.model:
            if isinstance(model, str):
                model = {"model": model}

            model_spec = EngineSpec(model)
            models.append(self._load_model(model_spec))

        # Load the base model if mix type is "difference"
        if mix_type == "difference":
            if "base" not in mix:
                raise ValueError("Must provide mix.base for difference mix type")

            model = mix.get("base")
            if isinstance(model, str):
                model = {"model": model}

            model_spec = EngineSpec(model)
            models.append(self._load_model(model_spec))

        # Check the arguments are all the same type
        types = [type(model) for model in models]

        if not all_same(types):
            raise ValueError(
                f"All model types must match, got {[t.__name__ for t in types]}"
            )

        # Start the result with all the non-module members of the first ModelSet
        res = {
            key: value
            for key, value in models[0].items()
            if not isinstance(value, torch.nn.Module)
        }

        # Get the keys in the first set that are modules
        module_keys = {
            key
            for key, value in models[0].items()
            if isinstance(value, torch.nn.Module)
        }

        # And then mix all the modules
        for key in module_keys:
            fuzz_key = None
            for check in ("unet", "text_encoder"):
                if key.endswith("_" + check):
                    fuzz_key = check

            mix_models = [model.get(key, model.get(fuzz_key)) for model in models]

            if any([not x for x in mix_models]):
                raise ValueError(f"No equivalent for {key} in one of the mix models")

            res[key] = self._mix_models(mix_method, mix_models, alpha)

        # And done
        return ModelSet(res)

    def _load_modelset_from_ckpt(
        self,
        weight_path,
        ckpt_config,
        whitelist=None,
        blacklist=None,
        local_only=False,
        fp16=None,
        ignore_patterns=None,
        allow_patterns=None,
    ) -> ModelSet:
        safetensor_paths = glob.glob("*.safetensors", root_dir=weight_path)
        ckpt_paths = glob.glob("*.ckpt", root_dir=weight_path) + glob.glob(
            "*.pt", root_dir=weight_path
        )

        safetensor_paths = list(
            huggingface_hub.utils.filter_repo_objects(
                safetensor_paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if fp16 is None:
            fp16 = self.mode.fp16

        extra_kwargs: dict[str, Any] = dict(
            whitelist=whitelist,
            blacklist=blacklist,
            dtype=torch.float16 if fp16 else None,
        )

        if safetensor_paths:
            if len(safetensor_paths) > 1:
                raise EnvironmentError(
                    f"Folder contained {len(safetensor_paths)} .safetensors files, there must be at most one."
                )

            extra_kwargs["safetensors_path"] = os.path.join(
                weight_path, safetensor_paths[0]
            )

        elif ckpt_paths:
            if len(ckpt_paths) > 1:
                raise EnvironmentError(
                    f"Folder contained {len(ckpt_paths)} .ckpt files, there must be at most one."
                )

            extra_kwargs["checkpoint_path"] = os.path.join(weight_path, ckpt_paths[0])

        else:
            raise EnvironmentError(
                f"Folder did not contain a .safetensors or .ckpt file."
            )

        with ckpt_utils.local_only(local_only):
            models = ckpt_utils.load_as_models(ckpt_config, **extra_kwargs)

        for model in models.values():
            model._source = (
                f"Ckpt {safetensor_paths[0] if safetensor_paths else ckpt_paths[0]}"
            )

        return ModelSet(models)

    def _load_from_weights(
        self, spec: EngineSpec, weight_path: str, local_only=False
    ) -> ModelSet:
        fp16 = False if spec.fp16 == "prevent" else None

        kwargs = dict(
            local_only=local_only,
            fp16=fp16,
            ignore_patterns=spec.ignore_patterns,
            allow_patterns=spec.allow_patterns,
        )

        # A pipeline has a top-level json file that describes a set of models
        if spec.type == "pipeline":
            models = self._load_modelset_from_weights(
                weight_path,
                whitelist=spec.whitelist,
                blacklist=spec.blacklist,
                **kwargs,
            )
        elif spec.type.startswith("ckpt/"):
            ckpt_config = spec.type[len("ckpt/") :]
            models = self._load_modelset_from_ckpt(
                weight_path,
                ckpt_config,
                whitelist=spec.whitelist,
                blacklist=spec.blacklist,
                **kwargs,
            )
        # `clip` type is a special case that loads the same weights into two different models
        elif spec.type == "clip":
            models = ModelSet(
                {
                    "clip_model": self._load_model_from_weights(
                        weight_path, "clip_model", **kwargs
                    ),
                    "feature_extractor": self._load_model_from_weights(
                        weight_path, "feature_extractor", **kwargs
                    ),
                }
            )
        # Otherwise load the individual model
        else:
            models = ModelSet(
                {
                    spec.type: self._load_model_from_weights(
                        weight_path, spec.type, spec.class_name, **kwargs
                    )
                }
            )

        return models

    def _load_from_weight_candidates(self, spec: EngineSpec) -> ModelSet:
        candidates = self._get_weight_path_candidates(spec)

        failures = []

        def exception_to_str(e):
            return "".join(traceback.format_exception(e)) if IS_DEV else str(e)

        for callback, args, kwargs in candidates:
            weight_path = None
            try:
                weight_path = callback(spec, *args, **kwargs)
                return self._load_from_weights(
                    spec, weight_path, local_only=kwargs.get("local_only")
                )
            except ValueError as e:
                if (message := exception_to_str(e)) not in failures:
                    failures.append(message)
            except Exception as e:
                if weight_path:
                    errstr = (
                        f"Error when trying to load weights from {weight_path}. "
                        + exception_to_str(e)
                    )
                    if errstr not in failures:
                        failures.append(errstr)
                else:
                    raise e

        if spec.is_engine:
            name = f"engine {spec.id}"
        else:
            name = f"model {spec.model_id}"

        raise EnvironmentError(
            "\n  - ".join([f"Failed to load {name}. Failed attempts:"] + failures)
        )

    def _load_from_reference(self, spec: EngineSpec) -> ModelSet:
        modelid, *submodel = spec.model[1:].split("/")
        if submodel:
            if len(submodel) > 1:
                raise EnvironmentError(
                    f"Can't have multiple sub-model references ({modelid}/{'/'.join(submodel)})"
                )
            submodel = submodel[0]
        else:
            submodel = None

        # Let type system know submodule is now a string
        assert isinstance(submodel, str | None), f"submodel of type {type(submodel)}"

        # If we've previous loaded this model, just return the same model
        if modelid in self._models:
            model = self._models[modelid]

        else:
            logger.info(f"    - Model {modelid}...")

            # Otherwise find the specification that matches the model_id reference
            specs = [
                spec
                for spec in self.engines
                if spec.enabled and spec.is_model and spec.model_id == modelid
            ]

            if not specs:
                raise EnvironmentError(f"Model {modelid} referenced does not exist")

            # And load it, storing in cache before continuing
            self._models[modelid] = model = self._load_model(specs[0])
            model.freeze()

        if submodel:
            return ModelSet({submodel: getattr(model, submodel)})
        elif spec.whitelist or spec.blacklist:
            include = set(model.keys())
            if spec.whitelist:
                include = include & set(spec.whitelist)
            if spec.blacklist:
                include = include - set(spec.blacklist)

            return ModelSet({k: v for k, v in model.items() if k in include})
        else:
            return model.copy()

    def _load_model(self, spec: EngineSpec, overrides=True):
        try:
            # Call the correct subroutine based on source to build the model
            if spec.model_is_empty:
                model = ModelSet()
            elif spec.model_is_reference:
                model = self._load_from_reference(spec)
            elif spec.type == "mix":
                model = self._load_mixed_model(spec)
            else:
                model = self._load_from_weight_candidates(spec)

        except Exception as e:
            fallback = spec.fallback
            if fallback:
                fallback_spec = EngineSpec({"model": fallback})
                model = self._load_from_reference(fallback_spec)
            else:
                raise e

        assert isinstance(model, ModelSet), f"model from {spec._data} is not a ModelSet"

        if overrides and spec.overrides:
            for name, override in spec.overrides.items():
                if isinstance(override, str):
                    override = {"model": override}

                override_spec = EngineSpec({**override, "type": name})
                override_model = self._load_model(override_spec)

                if override_model.is_singular():
                    model[name] = override_model.first()
                else:
                    model.update(override_model)

        return model

    def _inspect_kwargs(self, callable):
        class_init_params = inspect.signature(callable).parameters
        regular_params = {
            k: v
            for k, v in class_init_params.items()
            if (v.kind is v.POSITIONAL_OR_KEYWORD or v.kind is v.KEYWORD_ONLY)
            and k != "self"
        }

        takes_kwargs = any(
            [p.kind is p.VAR_KEYWORD for p in class_init_params.values()]
        )

        expected = set(regular_params.keys())

        required = set(
            [
                name
                for name, param in regular_params.items()
                if param.default is inspect._empty
            ]
        )

        return expected, required, takes_kwargs

    def _instantiate_pipeline(self, engine, model, extra_kwargs):
        fqclass_name = engine.get("class", "UnifiedPipeline")
        fqclass_name, factory_name, args = self._parse_class_details(fqclass_name)
        class_obj = self._import_class(fqclass_name)

        available = set(model.keys())

        expected, required, takes_kwargs = self._inspect_kwargs(class_obj.__init__)
        required = required - {"safety_checker"} - args.keys()

        if required - available:
            raise EnvironmentError(
                "Model definition did not provide model(s) the pipeline requires. Missing: "
                + repr(required - available)
            )

        modules = {
            k: clone_model(model[k])
            for k in (available if takes_kwargs else expected & available)
        }

        if "safety_checker" in expected and "safety_checker" not in available:
            modules["safety_checker"] = None

        if False:
            # Debug print source of each model
            max_len = max([len(n) for n in modules.keys()])
            for n, m in modules.items():
                print(f"{n.rjust(max_len, ' ')} | {'None' if m is None else m._source}")

        if "hintset_manager" in expected and engine.hintset:
            extra_kwargs["hintset_manager"] = self._hintsets[engine.hintset]

        modules = {**args, **modules, **extra_kwargs}
        return class_obj(**modules)

    def _instantiate_wrapper(self, spec, pipeline, model):
        meta = pipeline_meta.get_meta(pipeline)

        if wrap_class_name := meta.get("wrapper"):
            wrap_class = self._import_class(wrap_class_name)
        elif isinstance(pipeline, DiffusionPipeline):
            wrap_class = DiffusionPipelineWrapper
        else:
            wrap_class = PipelineWrapper

        expected, required, takes_kwargs = self._inspect_kwargs(wrap_class.__init__)
        required = required - {"id", "mode", "pipeline"}

        modules = {
            k: clone_model(model[k])
            for k in (model.keys() if takes_kwargs else expected & model.keys())
        }

        return wrap_class(id=spec.id, mode=self._mode, pipeline=pipeline, **modules)

    def _build_pipeline_for_engine(self, spec: EngineSpec):
        model = self._engine_models.get(spec.id)
        if not model:
            raise EngineNotReadyError("Not ready yet")

        pipeline = self._instantiate_pipeline(spec, model, {})

        if spec.options:
            try:
                pipeline.set_options(spec.options)
            except Exception:
                raise ValueError(
                    f"Engine {spec.id} has options, but created pipeline rejected them"
                )

        return self._instantiate_wrapper(spec, pipeline, model)

    def _build_hintset(self, hintset_id, whitelist="*", with_models=True):
        if isinstance(whitelist, str):
            whitelist = [whitelist]

        hintset_spec = self._find_hintset_spec(hintset_id)
        if not hintset_spec:
            raise EnvironmentError(f"Hintset {hintset_id} not defined anywhere")

        result = {}
        for name, handler in hintset_spec.items():
            if name.startswith("@"):
                subhintset = self._build_hintset(
                    name[1:], whitelist=handler, with_models=with_models
                )
                result.update(subhintset)

            else:
                whitelisted = any((fnmatch(name, pattern) for pattern in whitelist))
                if not whitelisted:
                    continue

                spec = EngineSpec({"model": handler["model"]})

                aliases = handler.get("aliases", [])
                if isinstance(aliases, str):
                    aliases = [aliases]

                result[name] = dict(
                    name=name,
                    models=self._load_model(spec).as_dict() if with_models else None,
                    types=[name] + aliases,
                    priority=handler.get("priority", 100),
                )

        return result

    def loadPipelines(self):

        logger.info("Loading engines...")
        self.status = "loading"

        for engine in self.engines:
            if not engine.enabled:
                continue

            # If this isn't an engine (but a model, or a depth extractor, skip)
            if not engine.is_engine:
                continue

            engineid = engine.id
            if engine.default:
                self._defaults[engine.task] = engineid

            logger.info(f"  - Engine {engineid}...")

            self._engine_models[engineid] = self._load_model(engine)

            if engine.hintset and engine.hintset not in self._hintsets:
                hintset_id = engine.hintset
                logger.info(f"  - Hintset {hintset_id}...")

                self._hintsets[hintset_id] = hintset_manager = HintsetManager()

                for handler in self._build_hintset(hintset_id).values():
                    hintset_manager.add_hint_handler(**handler)

        if self.batchMode.autodetect:
            self.batchMode.run_autodetect(self)

        self.status = "ready"

    def _fixcfg(self, model, key, test, value):
        if hasattr(model.config, key) and test(getattr(model.config, key)):
            print("Fixing", model._source)
            new_config = dict(model.config)
            new_config[key] = value
            model._internal_dict = FrozenDict(new_config)

    def _save_model_as_safetensor(self, spec: EngineSpec):
        # What's the local model attribute in the spec?
        local_model_attr = "local_model_fp16" if self.mode.fp16 else "local_model"

        _id = spec.id if spec.id else spec.model_id
        type = spec.type
        outpath = spec.get(local_model_attr)

        if not outpath:
            raise EnvironmentError(
                f"Can't save safetensor for {type} {_id} if {local_model_attr} not set"
            )

        if not os.path.isabs(outpath):
            outpath = os.path.join(self._weight_root, outpath)

        print(f"Saving {type} {_id} to {outpath}")

        # Load the model
        # TODO: Prevent references, not overrides (override with explicit modle OK)
        models = self._load_model(spec, overrides=False)

        if type == "pipeline" or type.startswith("ckpt/"):
            for name, model in models.items():
                if not model:
                    continue

                # Fix model issues before saving
                if name == "scheduler":
                    self._fixcfg(model, "steps_offset", lambda x: x != 1, 1)
                elif name == "unet":
                    self._fixcfg(model, "sample_size", lambda x: x < 64, 64)

                subpath = os.path.join(outpath, name)
                print(f"  Submodule {name} to {subpath}")
                model.save_pretrained(save_directory=subpath, safe_serialization=True)

            config_path = os.path.join(outpath, "model_index.json")
            if not os.path.exists(config_path):
                cfg = DiffusionPipeline()
                cfg.register_modules(**{k: v for k, v in models.items() if v})
                cfg.to_json_file(config_path)

        elif type == "clip":
            models.clip_model.save_pretrained(
                save_directory=outpath, safe_serialization=True
            )

            inpath = models.clip_model.config._name_or_path

            if not os.path.samefile(inpath, outpath):
                for cfg_file in glob.glob(os.path.join(inpath, "*.json")):
                    shutil.copy(cfg_file, outpath)
        else:
            model = list(models.values())[0]
            model.save_pretrained(save_directory=outpath, safe_serialization=True)

    def _find_specs(
        self,
        id: str | Iterable[str] | None = None,
        model_id: str | Iterable[str] | None = None,
    ):
        if id and model_id:
            raise ValueError("Must provide only one of id or model_id")
        if not id and not model_id:
            raise ValueError("Must provide one of id or model_id")

        key = "id" if id else "model_id"
        val = id if id else model_id
        assert val
        val = (val,) if isinstance(val, str) else val

        return (
            spec
            for spec in self.engines
            if key in spec
            and any((True for pattern in val if fnmatch(spec.get(key), pattern)))
        )

    def _find_spec(
        self,
        id: str | Iterable[str] | None = None,
        model_id: str | Iterable[str] | None = None,
    ):
        res = self._find_specs(id=id, model_id=model_id)
        return next(res, None)

    def find_by_hint(self, hints: str | Iterable[str], task: str | None = None):
        if isinstance(hints, str):
            hints = (hints,)

        candidates = [
            spec
            for spec in self.engines
            if spec.enabled and spec.is_engine and (task is None or spec.task == task)
        ]

        for hint in hints:
            for spec in candidates:
                if hint in spec.id:
                    return spec.id

        return None

    def _find_hintset_spec(self, hintset_id) -> HintsetSpec | None:
        for hintset in self.hintsets:
            if hintset.hintset_id == hintset_id:
                return hintset

        return None

    def save_models_as_safetensor(self, patterns):
        specs = self._find_specs(model_id=patterns)

        for spec in specs:
            self._save_model_as_safetensor(spec)

        print("Done")

    def _find_referenced_weightspecs(self, spec: EngineSpec):
        referenced = []

        if spec.model_is_reference:
            model_id, *_ = spec.model[1:].split("/")
            model_spec = self._find_spec(model_id=model_id)
            referenced += self._find_referenced_weightspecs(model_spec)
        elif spec.type == "mix":
            mix_models = spec.model
            if spec.mix and spec.mix.get("base"):
                mix_models += [spec.mix.get("base")]

            for model in mix_models:
                if isinstance(model, str):
                    model = {"model": model}

                mix_spec = EngineSpec({**model})
                referenced += self._find_referenced_weightspecs(mix_spec)
        elif not spec.model_is_empty and not spec.type.startswith("ckpt/"):
            referenced.append(spec)

        if spec.overrides:
            for name, override in spec.overrides.items():
                if isinstance(override, str):
                    override = {"model": override}

                override_spec = EngineSpec({**override, "type": name})
                referenced += self._find_referenced_weightspecs(override_spec)

        return referenced

    def save_engine_as_safetensor(self, patterns):
        specs = self._find_specs(id=patterns)
        specs = [spec for spec in specs if spec.enabled]

        involved = []

        for spec in specs:
            involved += self._find_referenced_weightspecs(spec)

        unique = {
            f"e/{spec.id}" if spec.is_engine else f"m/{spec.model_id}": spec
            for spec in involved
        }

        for spec in unique.values():
            try:
                self._save_model_as_safetensor(spec)
            except Exception as e:
                print(
                    f"Skipping {spec.human_id}, error received trying to save. Error was {e}."
                )

        print("Done")

    def getStatus(self):
        return {
            engine.id: engine.id in self._engine_models
            for engine in self.engines
            if engine.enabled and engine.is_engine
        }

    def getStatusByID(self, engine_id):
        return engine_id in self._engine_models

    def _return_pipeline_to_pool(self, slot):
        assert slot.pipeline, "No pipeline to return to pool"

        # Get the current slot pipeline
        pipeline = slot.pipeline

        # Deactivate and remove it from the slot
        slot.pipeline.deactivate()
        slot.pipeline = None

        # Return it to the pool (creating a pool if needed)
        pool = self._available_pipelines.setdefault(pipeline.id, deque())
        pool.append(pipeline)

    def _get_pipeline_from_pool(self, slot, id):
        assert not slot.pipeline, "Cannot allocate pipeline to full device slot"

        # Get the pool. If none available, return
        pool = self._available_pipelines.get(id)
        if not pool:
            return None

        # Try getting a pipeline from the pool. Again, if none available, just return
        try:
            pipeline = pool.pop()
        except IndexError:
            return None

        # Assign the pipeline to the slot and activate
        slot.pipeline = pipeline
        slot.pipeline.activate(slot.device)

        return pipeline

    @contextmanager
    def with_engine(self, id=None, task=None):
        """
        Get and activate a pipeline
        TODO: Better activate / deactivate logic. Right now we just keep a max of one pipeline active.
        """

        if id is None:
            id = self._defaults[task if task else "generate"]

        if id is None:
            raise EngineNotFoundError("No engine ID provided and no default is set.")

        # Get the engine spec
        spec = self._find_spec(id=id)
        if not spec or not spec.enabled:
            raise EngineNotFoundError(f"Engine ID {id} doesn't exist or isn't enabled.")

        if task is not None and task != spec.task:
            raise ValueError(f"Engine ID {id} is for task '{spec.task}' not '{task}'")

        # Get device queue slot
        slot = self._device_queue.get()

        try:
            # Get pipeline (create if all pipelines for the id are busy)

            # If a pipeline is already active on this device slot, check if it's the right
            # one. If not, deactivate it and clear it
            if slot.pipeline and slot.pipeline.id != id:
                old_id = slot.pipeline.id
                self._return_pipeline_to_pool(slot)

                if self._ram_monitor:
                    print(f"Existing pipeline {old_id} deactivated")
                    self._ram_monitor.print()

            # If there's no pipeline on this device slot yet, find it (creating it
            # if all the existing pipelines are busy)
            if not slot.pipeline:
                existing = True
                self._get_pipeline_from_pool(slot, id)

                if not slot.pipeline:
                    existing = False
                    slot.pipeline = self._build_pipeline_for_engine(spec)
                    slot.pipeline.activate(slot.device)

                if self._ram_monitor:
                    print(
                        f"{'Existing' if existing else 'New'} pipeline {id} activated"
                    )
                    self._ram_monitor.print()

            # Do the work
            yield slot.pipeline
        finally:
            # Release device handle
            self._device_queue.put(slot)

        # All done
