# To trigger binding of Uniformer backbone into mmseg
import glob
import json
import os

import huggingface_hub
import mmcv
import safetensors
import torch
import yaml

from gyre import torch_safe_unpickler

CONFIG_PATTERN = ["*.yaml", "*.json", "*.py"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]


class SafeTupleLoader(yaml.SafeLoader):
    def tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeTupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", SafeTupleLoader.tuple)


class ModelLoaderBase:
    @classmethod
    def get_matching_path(cls, path, name, patterns, allow_patterns, ignore_patterns):
        paths = []
        for pattern in patterns:
            paths += glob.glob(pattern, root_dir=path)

        kwargs = {}
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns is not None:
            kwargs["ignore_patterns"] = ignore_patterns

        paths = list(huggingface_hub.utils.filter_repo_objects(paths, **kwargs))

        if not paths:
            raise FileNotFoundError(f"No {name} found at {path}")
        elif len(paths) > 1:
            raise FileNotFoundError(
                f"Too many posible {name}s found at {path}. "
                "Try adding an allow_patterns or ignore_patterns to your model spec."
            )

        return os.path.join(path, paths[0])

    @classmethod
    def load_config(cls, config_path):
        if config_path.endswith(".yaml"):
            config_str = open(config_path, "r").read()
            return yaml.load(config_str, Loader=SafeTupleLoader)

        raise RuntimeError(f"Don't know how to load config {config_path}")

    @classmethod
    def load_model(cls, model_path):
        if model_path.endswith(".safetensors"):
            safedata = safetensors.safe_open(model_path, framework="pt", device="cpu")
            if encoded_meta := safedata.metadata():
                meta = {k: json.loads(v) for k, v in encoded_meta.items()}
            else:
                meta = {}
            state_dict = {k: safedata.get_tensor(k) for k in safedata.keys()}
        else:
            state_dict = torch.load(model_path, pickle_module=torch_safe_unpickler)

            if "meta" in state_dict and "state_dict" in state_dict:
                meta, state_dict = state_dict["meta"], state_dict["state_dict"]
            else:
                meta = {}

        return meta, state_dict
