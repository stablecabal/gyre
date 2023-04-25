import logging
from copy import deepcopy
from functools import partial
from itertools import groupby
from typing import Sized

import torch

from gyre.pipeline.model_utils import (
    ModelHook,
    add_hook,
    get_hook,
    get_hooks,
    remove_hook,
)

logger = logging.getLogger(__name__)

# --------------
# From this line to marker, taken from https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris/blob/main/lycoris.py
# Used under Apache 2.0 license
# Changes:
# - Automatic reformatting
# --------------


class LycoUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


class LycoHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None


class FullModule:
    def __init__(self):
        self.weight = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.shape = None


class IA3Module:
    def __init__(self):
        self.w = None
        self.alpha = None
        self.scale = None
        self.dim = None
        self.on_input = None


class LycoKronModule:
    def __init__(self):
        self.w1 = None
        self.w1a = None
        self.w1b = None
        self.w2 = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self._alpha = None
        self.scale = None
        self.dim = None
        self.shape = None
        self.bias = None

    @property
    def alpha(self):
        if self.w1a is None and self.w2a is None:
            return None
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, x):
        self._alpha = x


def make_weight_cp(t, wa, wb):
    temp = torch.einsum("i j k l, j r -> i r k l", t, wb)
    return torch.einsum("i j k l, i r -> r j k l", temp, wa)


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)


def _rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def _rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum("n m k l, i n, m j -> i j k l", mid, up, down)


def rebuild_weight(
    module, orig_weight: torch.Tensor, dyn_dim: int = None
) -> torch.Tensor:
    output_shape: Sized
    if module.__class__.__name__ == "LycoUpDownModule":
        up = module.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = module.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [up.size(0), down.size(1)]
        if (mid := module.mid_model) is not None:
            # cp-decomposition
            mid = mid.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = _rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = _rebuild_conventional(up, down, output_shape, dyn_dim)

    elif module.__class__.__name__ == "LycoHadaModule":
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]

        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape)

        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape)

        updown = updown1 * updown2

    elif module.__class__.__name__ == "FullModule":
        output_shape = module.weight.shape
        updown = module.weight.to(orig_weight.device, dtype=orig_weight.dtype)

    elif module.__class__.__name__ == "IA3Module":
        output_shape = [module.w.size(0), orig_weight.size(1)]
        if module.on_input:
            output_shape.reverse()
        else:
            module.w = module.w.reshape(-1, 1)
        updown = orig_weight * module.w

    elif module.__class__.__name__ == "LycoKronModule":
        if module.w1 is not None:
            w1 = module.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b

        if module.w2 is not None:
            w2 = module.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        elif module.t2 is None:
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = make_weight_cp(t2, w2a, w2b)

        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        updown = make_kron(output_shape, w1, w2)

    else:
        raise NotImplementedError(
            f"Unknown module type: {module.__class__.__name__}\n"
            "If the type is one of "
            "'LycoUpDownModule', 'LycoHadaModule', 'FullModule', 'IA3Module', 'LycoKronModule'"
            "You may have other lyco extension that conflict with locon extension."
        )

    if hasattr(module, "bias") and module.bias != None:
        updown = updown.reshape(module.bias.shape)
        updown += module.bias.to(orig_weight.device, dtype=orig_weight.dtype)
        updown = updown.reshape(output_shape)

    if len(output_shape) == 4:
        updown = updown.reshape(output_shape)

    if orig_weight.size().numel() == updown.size().numel():
        updown = updown.reshape(orig_weight.shape)
    # print(torch.sum(updown))
    return updown


# --------------
# End of copied section
# --------------


class LycorisHook(ModelHook):
    """
    Only a single LycorisHook should be attached to any model.
    The Lycoris library returns merged model weights, not additional weights.
    TODO: Maybe rewrite to work like the LoraHook does (once the upstream libray stabilises)
    """

    def __init__(self):
        self.lycorii = {}
        self.scales = {}
        self.dyn_dim = {}
        self.original_weight = None
        self.cached_weight = None

    def add_lycoris(self, id, lycoris):
        self.lycorii[id] = lycoris
        self.cached_weight = None

    def set_scale(self, id, weight):
        if id in self.lycorii:
            self.scales[id] = weight
            self.cached_weight = None

    def set_dyn_dim(self, id, dyn_dim):
        if id in self.lycorii:
            self.dyn_dim[id] = dyn_dim
            self.cached_weight = None

    def init_hook(self, module):
        return module

    def _calc_updown(self, id, original_weight):
        lycoris = self.lycorii[id]
        updown = rebuild_weight(lycoris, original_weight)

        dim_candidates = []
        if self.dyn_dim.get(id):
            dim_candidates.append(self.dyn_dim.get(id))
        if lycoris.dim:
            dim_candidates.append(lycoris.dim)

        dim = min(dim_candidates) if dim_candidates else None

        scale = 1.0
        if lycoris.scale:
            scale = lycoris.scale
        elif dim is not None and lycoris.alpha is not None:
            scale = lycoris.alpha / dim

        return updown * scale * self.scales.get(id, 1.0)

    def pre_forward(self, module, *args, **kwargs):
        # Pre-forward, figure out and replace the weight for this module
        if module.weight.device == torch.device("meta"):
            raise RuntimeError(
                "Can't apply a Lycoris to a module that's on the meta device"
            )

        if self.original_weight is not None:
            raise RuntimeError("Original weight was already present on pre_forward")

        if self.cached_weight is not None:
            weight = self.cached_weight

        else:
            weight = deepcopy(module.weight)

            if isinstance(module, torch.nn.MultiheadAttention):
                raise NotImplementedError()

            for id in self.lycorii.keys():
                weight += self._calc_updown(id, module.weight)

            self.cached_weight = weight

        self.original_weight, module.weight = module.weight, weight
        return args, kwargs

    def post_forward(self, module, output):
        if self.original_weight is None:
            raise RuntimeError(
                "No original weight available to restore on post_forward."
            )

        # Post-forward, restore the original weight
        module.weight, self.original_weight = self.original_weight, None
        return output

    def detach_hook(self, module):
        # Make sure the weights are properly restored
        if self.original_weight is not None:
            if module.weight.device == torch.device("meta"):
                logger.debug(
                    "Original weight remained on detach, but model already on meta device"
                )
            else:
                module.weight, self.original_weight = self.original_weight, None

        return module


def remove_lycoris_from_module(module):
    remove_hook(module, LycorisHook)


def remove_lycoris_from_model(model):
    for module in model.modules():
        remove_lycoris_from_module(module)


def set_lycoris_scale(module, id, scale=1.0):
    for child in module.modules():
        for lycoris_hook in get_hooks(child, LycorisHook):
            lycoris_hook.set_scale(id, scale)


def _set_item(name, lycoris, tensor):
    setattr(lycoris, name, tensor.item())
    return tensor.item()


def _set_weight(name, lycoris, tensor):
    setattr(lycoris, name, tensor)
    return tensor


def _set_weight_and_dim(name, lycoris, tensor):
    tensor = _set_weight(name, lycoris, tensor)
    setattr(lycoris, "dim", tensor.shape[0])
    return tensor


# This function based on https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris/blob/main/lycoris.py#L353
# Changes: basically totally rewritten
def _set_model(orig_module, kind, name, lycoris, tensor):
    module = None

    if isinstance(
        orig_module,
        torch.nn.Linear
        | torch.nn.modules.linear.NonDynamicallyQuantizableLinear
        | torch.nn.MultiheadAttention,
    ):
        tensor = tensor.reshape(tensor.shape[0], -1)
        module = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)

    elif isinstance(orig_module, torch.nn.Conv2d):
        # Depending on if this is down, up or mid, the exact Conv2D structure might be one of two options
        matched_args = dict(
            kernel_size=orig_module.kernel_size,
            stride=orig_module.stride,
            padding=orig_module.padding,
        )

        internal_args = dict(
            kernel_size=(1, 1),
        )

        # Choose one of those two arg sets based on kind
        if kind == "down":
            if len(tensor.shape) == 2:
                tensor = tensor.reshape(tensor.shape[0], -1, 1, 1)

            if tensor.shape[2] != 1 or tensor.shape[3] != 1:
                kwargs = matched_args
            else:
                kwargs = internal_args
        elif kind == "mid":
            kwargs = matched_args
        elif kind == "up":
            kwargs = internal_args
        else:
            raise RuntimeError(f"Unknown kind passed to _set_model: {kind}")

        # And build the module
        module = torch.nn.Conv2d(tensor.shape[1], tensor.shape[0], **kwargs, bias=False)

    else:
        raise ValueError(f"Can't apply Lycoris to {type(orig_module).__name__}")

    with torch.no_grad():
        if tensor.shape != module.weight.shape:
            tensor = tensor.reshape(module.weight.shape)
        module.weight.copy_(tensor)

    setattr(lycoris, name, module)
    return tensor


def _set_model_and_dim(orig_module, kind, name, lycoris, tensor):
    tensor = _set_model(orig_module, kind, name, lycoris, tensor)
    setattr(lycoris, "dim", tensor.shape[0])
    return tensor


def apply_lycoris(lora, id, unet=None, text_encoder=None):
    keys = list(lora.keys())
    keys.sort()

    type_count = {}

    for module_key, parameter_keys in groupby(keys, lambda key: key.split(".")[0]):
        parameter_keys = {key.split(".", 1)[1] for key in parameter_keys}

        if module_key.startswith("lora_te_"):
            module_subkey = module_key[len("lora_te_") :]
            module_type = "TextEncoder"
            module = text_encoder

        elif module_key.startswith("lora_unet_"):
            module_subkey = module_key[len("lora_unet_") :]
            module_type = "Unet"
            module = unet

        else:
            raise ValueError(
                f"Unknown module key in Lycoris, don't know how to apply - {module_key}"
            )

        if module is None:
            continue

        # Greedily try and use up key
        parts = module_subkey.split("_")
        while parts:
            for l in range(len(parts), 0, -1):
                name = "_".join(parts[:l])
                child = getattr(module, name, False)
                if child:
                    module, parts = child, parts[l:]
                    break
            else:
                raise RuntimeError(
                    f"Couldn't find model for {module_key} when applying LoRA"
                )

        key_handlers = {
            "alpha": partial(_set_item, "alpha"),
            "scale": partial(_set_item, "scale"),
        }

        if any((k.startswith("hada") for k in parameter_keys)):
            lycoris = LycoHadaModule()

            key_handlers.update(
                {
                    "hada_w1_a": partial(_set_weight, "w1a"),
                    "hada_w1_b": partial(_set_weight_and_dim, "w1b"),
                    "hada_w2_a": partial(_set_weight, "w2a"),
                    "hada_w2_b": partial(_set_weight_and_dim, "w2b"),
                    "hada_t1": partial(_set_weight, "t1"),
                    "hada_t2": partial(_set_weight, "t2"),
                }
            )

        elif any((k.startswith("lokr") for k in parameter_keys)):
            lycoris = LycoKronModule()

            key_handlers.update(
                {
                    "lokr_w1": partial(_set_weight, "w1"),
                    "lokr_w1_a": partial(_set_weight, "w1a"),
                    "lokr_w1_b": partial(_set_weight_and_dim, "w1b"),
                    "lokr_w2": partial(_set_weight, "w2"),
                    "lokr_w2_a": partial(_set_weight, "w2a"),
                    "lokr_w2_b": partial(_set_weight_and_dim, "w2b"),
                    "lokr_t2": partial(_set_weight, "t2"),
                }
            )

        elif "weight" in parameter_keys:
            lycoris = IA3Module()

            key_handlers.update(
                {
                    "weight": partial(_set_weight, "weight"),
                    "on_input": partial(_set_weight, "on_input"),
                }
            )

        elif "diff" in parameter_keys:
            lycoris = FullModule()

            key_handlers.update(
                {
                    "diff": partial(_set_weight, "weight"),
                }
            )

        else:
            lycoris = LycoUpDownModule()

            key_handlers.update(
                {
                    "lora_up.weight": partial(_set_model, module, "up", "up_model"),
                    "dyn_up": partial(_set_model, module, "down", "up_model"),
                    "lora_mid.weight": partial(_set_model, module, "mid", "mid_model"),
                    "lora_down.weight": partial(
                        _set_model_and_dim, module, "down", "down_model"
                    ),
                    "dyn_down": partial(_set_model_and_dim, module, "up", "down_model"),
                }
            )

        # Pull out bias
        if hasattr(lycoris, "bias"):
            bias_keys = {"bias_indices", "bias_values", "bias_size"}
            if bias_keys & parameter_keys:
                if bias_keys - parameter_keys:
                    raise ValueError(
                        f"Some bias keys missing in Lycoris {bias_keys - parameter_keys}"
                    )

                lycoris.bias = torch.sparse_coo_tensor(
                    lora.get_tensor(f"{module_key}.bias_indices"),
                    lora.get_tensor(f"{module_key}.bias_values"),
                    tuple(lora.get_tensor(f"{module_key}.bias_size")),
                )

                parameter_keys -= bias_keys

        # Pull out all other keys
        for parameter_key in parameter_keys:
            if parameter_key not in key_handlers:
                raise ValueError("Don't know how to handle key {parameter_key}")

            key_handlers[parameter_key](
                lycoris, lora.get_tensor(f"{module_key}.{parameter_key}")
            )

        # Pull out shape
        if hasattr(lycoris, "shape") and hasattr(module, "weight"):
            lycoris.shape = module.weight.shape

        lycoris_hook = get_hook(module, LycorisHook)
        if lycoris_hook is False:
            lycoris_hook = LycorisHook()
            add_hook(module, lycoris_hook)

        lycoris_hook.add_lycoris(id, lycoris)

        lycoris_type = module_type + ":" + type(lycoris).__name__
        type_count[lycoris_type] = type_count.get(lycoris_type, 0) + 1

    # Just before we're done, log all the types we encountered
    logger.info(f"Lycoris types: {type_count}")
