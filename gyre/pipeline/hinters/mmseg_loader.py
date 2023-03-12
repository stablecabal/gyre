# To trigger binding of Uniformer backbone into mmseg
import glob
import json
import os

import huggingface_hub
import safetensors
import torch
from mmseg.apis import init_segmentor

from gyre import torch_safe_unpickler

# Neccesary so mmseg will pick up the extra backbone
from gyre.pipeline.hinters.models.uniformer import UniFormer


class MmsegLoader:
    @classmethod
    def load(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        config_paths = []
        for pattern in ["*.py"]:
            config_paths += glob.glob(pattern, root_dir=path)

        config_paths = list(
            huggingface_hub.utils.filter_repo_objects(
                config_paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if not config_paths:
            raise RuntimeError(f"No configs for mmseg found at {path}")
        elif len(config_paths) > 1:
            raise RuntimeError(
                f"Too many posible configs found for mmseg at {path}. "
                "Try adding an allow_patterns or ignore_patterns to your model spec."
            )

        config_path = os.path.join(path, config_paths[0])

        segmentor = init_segmentor(config_path)

        model_paths = []
        for pattern in ["*.safetensors", "*.pt", "*.pth"]:
            model_paths += glob.glob(pattern, root_dir=path)

        model_paths = list(
            huggingface_hub.utils.filter_repo_objects(
                model_paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if not model_paths:
            raise RuntimeError(
                f"No model found for {segmentor.__class__.__name__} at {path}"
            )
        elif len(model_paths) > 1:
            raise RuntimeError(
                f"Too many posible models found for {segmentor.__class__.__name__} at {path}. "
                "Try adding an allow_patterns or ignore_patterns to your model spec."
            )

        model_path = os.path.join(path, model_paths[0])

        if model_path.endswith(".safetensors"):
            safedata = safetensors.safe_open(model_path, framework="pt", device="cpu")
            meta = {k: json.loads(v) for k, v in safedata.metadata().items()}
            state_dict = {k: safedata.get_tensor(k) for k in safedata.keys()}
        else:
            state_dict = torch.load(model_path, pickle_module=torch_safe_unpickler)
            meta, state_dict = state_dict["meta"], state_dict["state_dict"]

        segmentor.load_state_dict(state_dict)
        segmentor.CLASSES = meta["CLASSES"]
        segmentor.PALETTE = meta["PALETTE"]

        # if torch_dtype != "auto":
        #     segmentor.to(torch_dtype)

        segmentor.eval()
        return segmentor
