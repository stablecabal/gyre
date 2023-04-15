from types import SimpleNamespace as SN

from gyre.pipeline.hinters.baenormal.NNET import NNET
from gyre.pipeline.model_loader_base import ModelLoaderBase

CONFIG_PATTERN = ["*.yaml"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]

DEFAULT_CONFIG = dict(
    mode="client",
    architecture="BN",
    pretrained="scannet",
    sampling_ratio=0.4,
    importance_ratio=0.7,
)


class BaenormalLoader(ModelLoaderBase):
    @classmethod
    def load(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        **config_overrides,
    ):
        config = {**DEFAULT_CONFIG}
        config.update(config_overrides)

        try:
            config_path = cls.get_matching_path(
                path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
            )
            config.update(cls.load_config(config_path))

        except FileNotFoundError:
            pass

        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        _, state_dict = cls.load_model(model_path)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        prefix = "module."
        state_dict = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

        model = NNET(SN(**config))
        model.load_state_dict(state_dict)

        if torch_dtype != "auto":
            model.to(torch_dtype)

        model.eval()
        return model
