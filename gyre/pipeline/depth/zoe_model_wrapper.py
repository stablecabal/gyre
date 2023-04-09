import functools
import os
import traceback
import types
from copy import deepcopy

import safetensors
import torch
from midas.backbones.utils import activations
from midas.base_model import BaseModel
from midas.model_loader import load_model as load_midas_model
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype
from zoedepth.models import model_io
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.builder import build_model as build_zoe_model
from zoedepth.utils.config import get_config as get_zoe_config

from gyre.pipeline.depth.midas_model_wrapper import global_midas_lock
from gyre.pipeline.model_loader_base import ModelLoaderBase

MIDAS_MAPPING = {
    "DPT_BEiT_L_512": "dpt_beit_large_512",
    "DPT_BEiT_L_384": "dpt_beit_large_384",
}


def copy_func(f, closure=None):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__ if closure is None else closure,
    )
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def find_value(dict, val):
    for k, v in dict.items():
        if v is val:
            return k
    return None


class ZoeModelWrapper(torch.nn.Module, ModelLoaderBase):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def __deepcopy__(self, memo):
        # MiDaS has a shared dict that causes deepclone to fail. So we just unset it before clone...
        self.model.core.core.pretrained.activations = None
        try:
            copy = deepcopy(self.model, memo)
            # And then set it to the shared dict again afterwards.
            copy.core.core.pretrained.activations = activations

            # Deepcopy doesn't copy functions - that means any functions which are closures
            # will remain closed over the original class and not over the new class.

            # One effect of this is that Zoe's hooks don't work. Zoe attached forward_hooks to
            # store submodule results into a dict - but in copies, the function will still store
            # the result in the _original_ dict. So we have to find closures and update their references

            # We cache the found closure references, because they tend to be repeatedly frequently.
            # Unforunately, because they aren't hashable, we have to use a list and search by `is` manually.
            cache = []

            for module_name, module in copy.named_modules():
                for hookkey, hook in module._forward_hooks.items():

                    replacements = []
                    for cell_name, cell in zip(
                        hook.__code__.co_freevars, hook.__closure__
                    ):
                        cell_value = cell.cell_contents
                        source_name = attr_name = None

                        # Ignore any closure'd non-mutable types (theoretically not safe, but so far fine)
                        if isinstance(cell_value, str | int | float):
                            replacements.append(cell)
                            continue

                        # Check in cache
                        if source_name is None:
                            for item, name, k in cache:
                                if item is cell_value:
                                    source_name, attr_name = name, k
                                    break

                        # Search through every module in network
                        if source_name is None:
                            for name, orig_module in self.model.named_modules():
                                k = find_value(orig_module.__dict__, cell_value)
                                if k is not None:
                                    cache.append((cell.cell_contents, name, k))
                                    source_name, attr_name = name, k
                                    break

                        if source_name is not None:
                            replacements.append(
                                types.CellType(
                                    getattr(copy.get_submodule(source_name), attr_name)
                                )
                            )
                        else:
                            raise ValueError(
                                f"Found mutable closure variable {cell_name} in hook on module {module_name} "
                                "but couldn't find the reference source. Deepcopy likely to fail."
                            )

                    copied_func = copy_func(hook, closure=tuple(replacements))
                    module._forward_hooks[hookkey] = copied_func

        finally:
            # Don't forget to put the activations dict back into the source
            self.model.core.core.pretrained.activations = activations

        return ZoeModelWrapper(copy)

    def infer(
        self, x, pad_input: bool = True, with_flip_aug: bool = True, **kwargs
    ) -> torch.Tensor:
        # Just run the model
        with global_midas_lock:
            return self.model.infer(x, pad_input, with_flip_aug, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        config_id="zoedepth",
        **config,
    ):
        # Get the config and find the model state names for each of Midas and Zoe
        config_kwargs = {}
        if "/" in config_id:
            config_id, config_version = config_id.split("/")
            config_kwargs["config_version"] = config_version

        config = get_zoe_config(config_id, "infer", **config_kwargs)

        midas_type = MIDAS_MAPPING.get(config["midas_model_type"])
        if midas_type is None:
            raise ValueError(f"No mapping for MiDaS tyoe {config['midas_model_type']}")

        zoe_name = config["pretrained_resource"].split("/")[-1]
        zoe_name, _ = os.path.splitext(zoe_name)

        # Find the matching state paths

        midas_path = cls.get_matching_path(
            path,
            "midas",
            [f"{midas_type}.safetensors", f"{midas_type}.pt"],
            allow_patterns,
            ignore_patterns,
        )

        zoe_path = cls.get_matching_path(
            path,
            "zoe",
            [f"{zoe_name}.safetensors", f"{zoe_name}.pt"],
            allow_patterns,
            ignore_patterns,
        )

        # Load Midas

        def load(self, path):
            if path.endswith(".safetensors"):
                safedata = safetensors.safe_open(path, framework="pt", device="cpu")
                parameters = {k: safedata.get_tensor(k) for k in safedata.keys()}
            else:
                parameters = torch.load(path, map_location="cpu")

            if "optimizer" in parameters:
                parameters = parameters["model"]

            self.load_state_dict(parameters)

        orig_load, BaseModel.load = BaseModel.load, load
        try:
            midas_model, *_ = load_midas_model("cpu", midas_path, midas_type)
        finally:
            BaseModel.load = orig_load

        # Load Zoe

        # Override build model that would otherwise re-download midas somewhere else
        def build(
            midas_model_type="DPT_BEiT_L_384",
            train_midas=False,
            use_pretrained_midas=True,
            fetch_features=False,
            freeze_bn=True,
            force_keep_ar=False,
            force_reload=False,
            **kwargs,
        ):
            if "img_size" in kwargs:
                kwargs = MidasCore.parse_img_size(kwargs)
            img_size = kwargs.pop("img_size", [384, 384])

            kwargs.update({"keep_aspect_ratio": force_keep_ar})
            midas_core = MidasCore(
                midas_model,
                trainable=train_midas,
                fetch_features=fetch_features,
                freeze_bn=freeze_bn,
                img_size=img_size,
                **kwargs,
            )
            midas_core.set_output_channels(midas_model_type)
            return midas_core

        def load_wts(model, checkpoint_path):
            if checkpoint_path.endswith(".safetensors"):
                safedata = safetensors.safe_open(
                    checkpoint_path, framework="pt", device="cpu"
                )
                state = {"model": {k: safedata.get_tensor(k) for k in safedata.keys()}}
            else:
                state = torch.load(checkpoint_path, map_location="cpu")

            return model_io.load_state_dict(model, state)

        orig_build, MidasCore.build = MidasCore.build, build
        orig_load_wts, model_io.load_wts = model_io.load_wts, load_wts
        try:
            config["pretrained_resource"] = "local::" + zoe_path
            zoe_model = build_zoe_model(config)
        finally:
            MidasCore.build = orig_build
            model_io.load_wts = orig_load_wts

        # Build wrapper model & return

        result = cls(zoe_model)
        # Zoe will just NaN if you try to run it in float16
        # result.to(torch_dtype)
        result.eval()

        return result
