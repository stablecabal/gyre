import functools
import inspect
from types import SimpleNamespace as SN

import segment_anything

build_sam = inspect.getmodule(segment_anything.build_sam)

from gyre.pipeline.model_loader_base import ModelLoaderBase

CONFIG_PATTERN = ["*.yaml"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]

SAMHQ_CONFIGS = {}


def extract_configs():
    def capture_config(dest, **kwargs):
        dest.update(kwargs)

    _build_sam_orig = build_sam._build_sam

    try:
        for name, builder in build_sam.sam_model_registry.items():
            config = SAMHQ_CONFIGS.setdefault(name, {})
            build_sam._build_sam = functools.partial(capture_config, dest=config)
            builder()
    finally:
        build_sam._build_sam = _build_sam_orig


extract_configs()


class SamHQLoader(ModelLoaderBase):
    @classmethod
    def load(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        model_base="default",
        **config_overrides,
    ):
        config = SAMHQ_CONFIGS[model_base]
        config.update(config_overrides)

        # Load any override config
        try:
            config_path = cls.get_matching_path(
                path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
            )
            config.update(cls.load_config(config_path))

        except FileNotFoundError:
            pass

        # Load the actual state
        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        _, state_dict = cls.load_model(model_path)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        sam = build_sam._build_sam(**config)

        # TODO: This is commented out in build_sam - wonder why
        # sam.eval()

        sam.load_state_dict(state_dict, strict=False)

        if torch_dtype != "auto":
            sam.to(torch_dtype)

        return sam
