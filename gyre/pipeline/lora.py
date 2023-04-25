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

    # First check - find any key to indicate a basic type

    detect_keys: dict[str, lora_type] = {
        ":0:up": "cloneofsimo",
        ".lora_up.weight": "kohya-ss",
        ".to_k_lora.up.weight": "diffusers",
    }

    def basic_type():
        for key in lora.keys():
            for pattern, type in detect_keys.items():
                if key.endswith(pattern):
                    return type

        raise ValueError("Unknown LoRA format (or not a LoRA)")

    type = basic_type()

    # Second check - for kohya-ss make sure there are only Lora (and not Lycoris) fields

    kohya_keys = (".lora_up.weight", ".lora_down.weight", ".alpha")

    if type == "kohya-ss":
        for key in lora.keys():
            for pattern in kohya_keys:
                if key.endswith(pattern):
                    break
            else:
                raise ValueError("LoRA contains unknown fields, probably a Lycoris")

    # OK, if we didn't throw an error, we're done

    return type


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

            r = self._r
            in_dim = module.in_channels
            out_dim = module.out_channels
            kernel = module.kernel_size
            stride = module.stride
            padding = module.padding

            self._lora_down = nn.Conv2d(in_dim, r, kernel, stride, padding, bias=False)
            self._lora_up = nn.Conv2d(r, out_dim, (1, 1), (1, 1), bias=False)

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

        result = (
            self._lora_up(self._lora_down(self._input)) * self._iscale * self._scale
        )

        return output + result

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


def remove_lora_from_model(model):
    for module in model.modules():
        remove_lora_from_module(module)


def set_lora_scale(module, id, scale=1.0):
    for child in module.modules():
        for lora_hook in get_hooks(child, LoraHook):
            if lora_hook.id == id:
                lora_hook.scale = scale


def apply_lora(lora, id, unet=None, text_encoder=None):
    # remove_lora_from_pipe(pipe)

    lora_type = detect_lora_type(lora)

    if lora_type == "cloneofsimo":
        apply_cloneofsimo_to_pipe(lora, id, unet=unet, text_encoder=text_encoder)
    elif lora_type == "diffusers":
        apply_diffusers_to_pipe(lora, id, unet=unet, text_encoder=text_encoder)
    elif lora_type == "kohya-ss":
        apply_kohya_to_pipe(lora, id, unet=unet, text_encoder=text_encoder)


def apply_cloneofsimo_to_pipe(lora, id, unet=None, text_encoder=None):
    loras = parse_safeloras(lora)

    logger.debug(f"Loading cloneofsimo Lora with {loras.keys()} models")

    for name, (lora, ranks, target) in loras.items():
        if name == "unet":
            model = unet
        elif name == "text_encoder":
            model = text_encoder
        else:
            raise ValueError(f"Unknown model in CloneOfSimo LoRA - {name}")

        if model is None:
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


def apply_diffusers_to_pipe(lora, id, unet=None, text_encoder=None):
    wrapped = _Wrapper(lora)

    # TODO: Find an example of a diffusers lora with text encoder LoRA
    logger.debug(f"Loading diffusers Lora (unet only)")

    if unet is None:
        return

    for key in wrapped.keys():
        if not key.endswith(".down.weight"):
            continue

        module = unet

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


def apply_kohya_to_pipe(lora, id, unet=None, text_encoder=None):
    wrapped = _Wrapper(lora)

    def has_keys_like(prep):
        return any((x.startswith(prep) for x in wrapped.keys()))

    applying = []

    if has_keys_like("lora_unet_") and unet is not None:
        applying.append("unet")

    if has_keys_like("lora_te_") and text_encoder is not None:
        applying.append("text_encoder")

    if applying:
        logger.debug(f"Loading kohya-ss Lora ({', '.join(applying)})")

    for key in wrapped.keys():
        if not key.endswith(".lora_down.weight"):
            continue

        if key.startswith("lora_te_"):
            module_key = key[len("lora_te_") :]
            module = text_encoder

        elif key.startswith("lora_unet_"):
            module_key = key[len("lora_unet_") :]
            module = unet

        else:
            raise ValueError(
                f"Unknown key in Kohya LoRA, don't know how to apply - {key}"
            )

        if module is None:
            continue

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
