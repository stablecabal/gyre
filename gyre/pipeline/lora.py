import logging
import re
import sys
import types
from typing import Literal

from safetensors import safe_open
from torch import nn

from gyre.pipeline.model_utils import (
    ModelHook,
    add_hook,
    get_hook,
    get_hooks,
    remove_hook,
)

# Nasty hack - detect if fire is installed, and if not, prevent lora import from failing

try:
    import fire
except ImportError:
    sys.modules["fire"] = types.ModuleType("fire")

from gyre.src.lora.lora_diffusion.lora import (
    _find_modules,
    parse_safeloras,
    parse_safeloras_embeds,
)

logger = logging.getLogger(__name__)

lora_type = Literal["cloneofsimo", "kohya-ss", "diffusers"]
# Notes:
# - diffusers only supports unet, not text_encoder


class _Wrapper:
    def __init__(self, lora: dict | safe_open):
        self._lora = lora

    def keys(self):
        return self._lora.keys()

    def tensor(self, key, default=None):
        if key not in self.keys():
            return default

        if isinstance(self._lora, dict):
            return self._lora[key]
        else:
            return self._lora.get_tensor(key)

    def items(self):
        for key in self.keys():
            yield key, self.tensor(key)


def detect_lora_type(lora) -> lora_type:
    """
    What format is this LoRA?
    """

    # Note: we assume a unet will always be included in the LORA

    detect_keys: dict[str, lora_type] = {
        "unet:0:up": "cloneofsimo",
        "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": "kohya-ss",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_k_lora.up.weight": "diffusers",
    }

    for k, t in detect_keys.items():
        # Needs to be .keys(), since lora might be a Safetensor not a Dict
        if k in lora.keys():
            return t

    raise ValueError("Unknown LoRA format (or not a LoRA)")


class LoraHook(ModelHook):
    """
    We use an accelerate hook instead of cloneofsimo/LoraInjected types to work even when
    unet or text_encoder has a CloneToGPUHook attached.
    """

    def __init__(self, id, up_weight, down_weight, r=4, alpha=None, scale=1.0):
        self.id = id
        self._up_weight = up_weight
        self._down_weight = down_weight
        self._r = r
        self._iscale = alpha / r if alpha else 1.0
        self._scale = scale

    def init_hook(self, module):
        if isinstance(module, nn.Conv2d):
            assert len(self._up_weight.shape) == len(self._down_weight.shape) == 4

            in_dim = module.in_channels
            out_dim = module.out_channels
            self._lora_down = nn.Conv2d(in_dim, self._r, (1, 1), bias=False)
            self._lora_up = nn.Conv2d(self._r, out_dim, (1, 1), bias=False)

        elif isinstance(module, nn.Linear):
            assert len(self._up_weight.shape) == len(self._down_weight.shape) == 2

            in_dim = module.in_features
            out_dim = module.out_features
            self._lora_down = nn.Linear(in_dim, self._r, bias=False)
            self._lora_up = nn.Linear(self._r, out_dim, bias=False)

        else:
            raise ValueError(
                f"Can't apply LoRA to module of type {module.__class__.__name__}"
            )

        self._lora_down.weight = nn.Parameter(self._down_weight)
        self._lora_up.weight = nn.Parameter(self._up_weight)

        return module

    def pre_forward(self, module, *args, **kwargs):
        self._input = kwargs["input"] if "input" in kwargs else args[0]
        return args, kwargs

    def post_forward(self, module, output):
        self._lora_down.to(self._input.device, self._input.dtype)
        self._lora_up.to(self._input.device, self._input.dtype)

        return (
            output
            + self._lora_up(self._lora_down(self._input)) * self._iscale * self._scale
        )

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value


def apply_lora_to_module(module, id, up_weight, down_weight, r, alpha=None):
    add_hook(
        module,
        LoraHook(id=id, up_weight=up_weight, down_weight=down_weight, r=r, alpha=alpha),
    )


def remove_lora_from_module(module):
    remove_hook(module, LoraHook)


def remove_lora_from_pipe(pipe):
    for module in pipe.text_encoder.modules():
        remove_lora_from_module(module)
    for module in pipe.unet.modules():
        remove_lora_from_module(module)


def set_lora_scale(module, id, scale=1.0):
    for child in module.modules():
        for lora_hook in get_hooks(child, LoraHook):
            if lora_hook.id == id:
                lora_hook.scale = scale


def apply_lora_to_pipe(pipe, lora, id):
    # remove_lora_from_pipe(pipe)

    lora_type = detect_lora_type(lora)

    if lora_type == "cloneofsimo":
        apply_cloneofsimo_to_pipe(pipe, lora, id)
    elif lora_type == "diffusers":
        apply_diffusers_to_pipe(pipe, lora, id)
    elif lora_type == "kohya-ss":
        apply_kohya_to_pipe(pipe, lora, id)


def apply_cloneofsimo_to_pipe(pipe, lora, id):
    loras = parse_safeloras(lora)

    logger.debug(f"Loading cloneofsimo Lora with {loras.keys()} models")

    for name, (lora, ranks, target) in loras.items():
        model = getattr(pipe, name, None)

        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue

        for _module, name, _child_module in _find_modules(
            model,
            target,
            search_class=[nn.Linear, nn.Conv2d],
        ):
            r = ranks.pop(0) if isinstance(ranks, list) else ranks
            up_weight = lora.pop(0)
            down_weight = lora.pop(0)

            apply_lora_to_module(
                _child_module, id=id, up_weight=up_weight, down_weight=down_weight, r=r
            )


def apply_diffusers_to_pipe(pipe, lora, id):
    wrapped = _Wrapper(lora)

    logger.debug(f"Loading diffusers Lora (unet only)")

    for key in wrapped.keys():
        if not key.endswith(".down.weight"):
            continue

        module = pipe.unet

        # Adjust the key to reference the Linear, rather than the Processor
        fixed_key = re.sub(
            r"processor.(.+)_lora.down.weight$",
            lambda match: match[1] + ".0" if match[1] == "to_out" else match[1],
            key,
        )

        # Find the linear or conv2d to apply it to
        for part in fixed_key.split("."):
            module = getattr(module, part)

        down_weight = wrapped.tensor(key)
        up_weight = wrapped.tensor(key.replace(".down.weight", ".up.weight"))
        r = down_weight.size()[0]

        apply_lora_to_module(
            module, id=id, up_weight=up_weight, down_weight=down_weight, r=r
        )


def apply_kohya_to_pipe(pipe, lora, id):
    wrapped = _Wrapper(lora)

    has_unet = any((x.startswith("lora_unet_") for x in wrapped.keys()))
    has_te = any((x.startswith("lora_te_") for x in wrapped.keys()))

    logger.debug(
        f"Loading kohya-ss Lora ({'unet' if has_unet else ''} {'text_encoder' if has_te else ''})"
    )

    for key in wrapped.keys():
        if not key.endswith(".lora_down.weight"):
            continue

        if key.startswith("lora_te_"):
            module_key = key[len("lora_te_") :]
            module = pipe.text_encoder

        elif key.startswith("lora_unet_"):
            module_key = key[len("lora_unet_") :]
            module = pipe.unet

        else:
            raise RuntimeError(f"Don't know pipe model to apply {key} to")

        # Greedily try and use up key
        parts = module_key.split(".")[0].split("_")
        while parts:
            for l in range(len(parts), 0, -1):
                name = "_".join(parts[:l])
                child = getattr(module, name, False)
                if child:
                    module, parts = child, parts[l:]
                    break
            else:
                raise RuntimeError(f"Couldn't find model for {key} when applying LoRA")

        down_weight = wrapped.tensor(key)
        up_weight = wrapped.tensor(key.replace(".lora_down.", ".lora_up."))
        alpha = wrapped.tensor(key.replace(".lora_down.weight", ".alpha"))
        r = down_weight.size()[0]

        apply_lora_to_module(
            module,
            id=id,
            up_weight=up_weight,
            down_weight=down_weight,
            r=r,
            alpha=alpha,
        )
