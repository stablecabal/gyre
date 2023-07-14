import importlib
import logging
from types import SimpleNamespace

from gyre.pipeline.model_loader_base import ModelLoaderBase
from gyre.pipeline.inpainting.lsm_hawp.detector import WireframeDetector

logger = logging.getLogger(__name__)


CONFIG_PATTERN = ["*.yaml", "*.yml"]
MODELS_PATTERN = ["*.safetensors", "*.ckpt", "*.pt", "*.pth"]


def with_prefix(patterns, prefix):
    return {f"{prefix}{pattern}" for pattern in patterns}


def as_sn(item):
    if type(item) is dict:
        return SimpleNamespace(**{k: as_sn(v) for k, v in item.items()})
    elif type(item) is list:
        return [as_sn(v) for v in item]
    else:
        return item


class ZitsPPLoader(ModelLoaderBase):
    @classmethod
    def class_from_fqname(cls, fqclass_name: str):
        *library_name, class_name = fqclass_name.split(".")
        library_name = ".".join(library_name)

        library_name = library_name.replace(
            "networks.", "gyre.pipeline.inpainting.zitspp."
        )

        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name, None)

        if not class_obj:
            raise ValueError(f"{class_name} not found in module {library}")

        return class_obj

    @classmethod
    def load_wf(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        **config_overrides,
    ):
        # Load base config
        try:
            config_path = cls.get_matching_path(
                path,
                "lsm_hawp config",
                with_prefix(CONFIG_PATTERN, "best_lsm_hawp"),
                allow_patterns,
                ignore_patterns,
            )
            config = cls.load_config(config_path)

        except FileNotFoundError:
            raise ValueError("Config for ZitsPP not found")

        # Override with any overrides
        config.update(config_overrides)

        model_path = cls.get_matching_path(
            path,
            "lsm_hawp model",
            with_prefix(MODELS_PATTERN, "best_lsm_hawp"),
            allow_patterns,
            ignore_patterns,
        )
        _, state_dict = cls.load_model(model_path)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        wf = WireframeDetector(as_sn(config))
        wf.load_state_dict(state_dict, strict=True)
        wf.eval()

        return wf

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
        # Load base config
        try:
            config_path = cls.get_matching_path(
                path,
                "config",
                with_prefix(CONFIG_PATTERN, "last"),
                allow_patterns,
                ignore_patterns,
            )
            config = cls.load_config(config_path)

        except FileNotFoundError:
            raise ValueError("Config for ZitsPP not found")

        # Override with any overrides
        config.update(config_overrides)

        # Load the actual state
        model_path = cls.get_matching_path(
            path,
            "model",
            with_prefix(MODELS_PATTERN, "last"),
            allow_patterns,
            ignore_patterns,
        )
        _, state_dict = cls.load_model(model_path)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Now build the models
        structure_upsample = cls.class_from_fqname(config["structure_upsample_class"])()
        edgeline_tsr = cls.class_from_fqname(config["edgeline_tsr_class"])()
        grad_tsr = cls.class_from_fqname(config["grad_tsr_class"])()
        ftr = cls.class_from_fqname(config["g_class"])(config=config["g_args"])

        model = SimpleNamespace(
            structure_upsample=structure_upsample,
            edgeline_tsr=edgeline_tsr,
            grad_tsr=grad_tsr,
            ftr=ftr,
        )

        # Load the unified state dict into the various models
        key_tops = {x.split(".")[0] for x in state_dict.keys()}
        for top in key_tops:
            target = getattr(model, top, None)
            if not target:
                logger.debug(f"No such model top {top} in ZitsPP")
            else:
                top = top + "."
                sub_dict = {
                    k[len(top) :]: v for k, v in state_dict.items() if k.startswith(top)
                }
                target.load_state_dict(sub_dict)

        # Run eval on every model
        structure_upsample.requires_grad_(False).eval()
        edgeline_tsr.requires_grad_(False).eval()
        grad_tsr.requires_grad_(False).eval()
        model.ftr.eval()

        return model
