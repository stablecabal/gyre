# To trigger binding of Uniformer backbone into mmseg
import glob
import json
import os
import traceback

import huggingface_hub
import mmcv
import safetensors
import torch
import yaml
from mmdet.apis import init_detector
from mmdet.core import get_classes
from mmpose.apis import init_pose_model
from mmseg.apis import init_segmentor

from gyre import torch_safe_unpickler

# Neccesary so mmseg will pick up the extra backbone
from gyre.pipeline.hinters.models.uniformer import UniFormer

CONFIG_PATTERN = ["*.yaml", "*.json", "*.py"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]


class SafeTupleLoader(yaml.SafeLoader):
    def tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeTupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", SafeTupleLoader.tuple)


class MmLoader:
    @classmethod
    def get_matching_path(cls, path, name, patterns, allow_patterns, ignore_patterns):
        paths = []
        for pattern in patterns:
            paths += glob.glob(pattern, root_dir=path)

        paths = list(
            huggingface_hub.utils.filter_repo_objects(
                paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if not paths:
            raise RuntimeError(f"No {name} found at {path}")
        elif len(paths) > 1:
            raise RuntimeError(
                f"Too many posible {name}s found at {path}. "
                "Try adding an allow_patterns or ignore_patterns to your model spec."
            )

        return os.path.join(path, paths[0])

    @classmethod
    def load_config(cls, config_path):
        if config_path.endswith(".yaml"):
            config_str = open(config_path, "r").read()
            return mmcv.Config(yaml.load(config_str, Loader=SafeTupleLoader))

        return config_path

    @classmethod
    def load_model(cls, model_path):
        if model_path.endswith(".safetensors"):
            safedata = safetensors.safe_open(model_path, framework="pt", device="cpu")
            meta = {k: json.loads(v) for k, v in safedata.metadata().items()}
            state_dict = {k: safedata.get_tensor(k) for k in safedata.keys()}
        else:
            state_dict = torch.load(model_path, pickle_module=torch_safe_unpickler)
            meta, state_dict = state_dict["meta"], state_dict["state_dict"]

        return meta, state_dict

    @classmethod
    def load_mmseg(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        config_path = cls.get_matching_path(
            path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
        )
        segmentor = init_segmentor(cls.load_config(config_path))

        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        meta, state_dict = cls.load_model(model_path)

        segmentor.load_state_dict(state_dict)
        segmentor.CLASSES = meta["CLASSES"]
        segmentor.PALETTE = meta["PALETTE"]

        # if torch_dtype != "auto":
        #     segmentor.to(torch_dtype)

        segmentor.eval()
        return segmentor

    @classmethod
    def load_mmdet(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        config_path = cls.get_matching_path(
            path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
        )
        detector = init_detector(cls.load_config(config_path))

        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        meta, state_dict = cls.load_model(model_path)

        detector.load_state_dict(state_dict)
        if "CLASSES" in meta:
            detector.CLASSES = meta["CLASSES"]
        else:
            detector.CLASSES = get_classes("coco")

        # if torch_dtype != "auto":
        #     detector.to(torch_dtype)

        detector.eval()
        return detector

    @classmethod
    def load_mmpose(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        config_path = cls.get_matching_path(
            path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
        )
        pose_model = init_pose_model(cls.load_config(config_path))

        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        meta, state_dict = cls.load_model(model_path)

        pose_model.load_state_dict(state_dict)

        # if torch_dtype != "auto":
        #     detector.to(torch_dtype)

        pose_model.eval()
        return pose_model
