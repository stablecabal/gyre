import functools
import logging
import weakref
from copy import deepcopy
from typing import Literal

import torch
from accelerate.hooks import (
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    remove_hook_from_module,
)
from accelerate.utils import send_to_device, set_module_tensor_to_device

logger = logging.getLogger(__name__)


def model_size(model) -> int:
    total = 0

    for name, param in model.named_parameters(recurse=True):
        total += param.numel() * param.element_size()

    for name, buffer in model.named_buffers(recurse=True):
        total += buffer.numel() * buffer.element_size()

    return total


class CloneToGPUHook(ModelHook):
    def __init__(self, execution_device, activate_callback, params, buffers):
        self.execution_device = execution_device
        self.activate_callback = activate_callback
        self.params = params
        self.buffers = buffers

        # active is tri-state, None means unknown
        self.active: bool | None = None

    def pre_forward(self, module, *args, **kwargs):
        if self.active is True:
            return args, kwargs

        self.active = None

        self.activate_callback()

        dev = self.execution_device
        meta_device = torch.device("meta")

        for name, param in module.named_parameters(recurse=False):
            if param.device == meta_device:
                # explicitly copy, as set_module_tensor_to_device won't create
                # a copy if the device is already correct
                new_param = self.params[name].to(dev, copy=True)
                set_module_tensor_to_device(module, name, dev, new_param)

        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device == meta_device:
                new_buffer = self.buffers[name].to(dev, copy=True)
                set_module_tensor_to_device(module, name, dev, new_buffer)

        self.active = True

        # TODO: This is a very specific fix, should be replaced by
        # something more generic
        if isinstance(module, torch.nn.MultiheadAttention):
            out_proj = module.out_proj
            hook = get_hook(out_proj, CloneToGPUHook)
            hook.pre_forward(out_proj, [], {})

        return (
            send_to_device(args, dev),
            send_to_device(kwargs, dev),
        )

    def reset(self, model):
        if self.active is False:
            return

        self.active = None

        for name in self.params.keys():
            set_module_tensor_to_device(model, name, "meta")
        for name in self.buffers.keys():
            set_module_tensor_to_device(model, name, "meta")

        self.active = False

        if isinstance(model, torch.nn.MultiheadAttention):
            out_proj = model.out_proj
            hook = get_hook(out_proj, CloneToGPUHook)
            hook.reset(out_proj)


class GPUExclusionSet:
    def __init__(self, name: str | None = None, max_activated=-1, mem_limit=-1):
        self.name = name
        self.tops: list[weakref.ref] = []
        self.activated: list[tuple[weakref.ref, int]] = []
        self.max_activated = max_activated
        self.mem_limit = mem_limit

    def add(self, top):
        self.tops.append(weakref.ref(top, self.clean_sets))

    def clean_sets(self, _):
        self.tops = [topref for topref in self.tops if topref() is not None]

    def reset(self, exclude: list[weakref.ref] | None = None):
        exclude = [] if exclude is None else list(exclude)

        for topref in self.tops:
            if topref in exclude:
                continue

            if (top := topref()) is None:
                continue

            for _, module in top.named_modules():
                hook = get_hook(module, CloneToGPUHook)
                if hook is not False:
                    hook.reset(module)

    def activate_for(self, top):
        topref = weakref.ref(top)
        return functools.partial(self.activate, topref, model_size(top))

    def activate(self, topref, size):
        # No-op if top is already the most recently activated
        if self.activated and self.activated[0][0] is topref:
            return

        cur_activated = [*self.activated]
        new_activated = [(topref, size)]
        total = size

        while cur_activated:
            othertopref, othersize = cur_activated[0]

            if othertopref is topref or othertopref() is None:
                cur_activated.pop(0)
                continue

            if self.max_activated > 0 and len(new_activated) >= self.max_activated:
                break

            if self.mem_limit > 0 and (total + cur_activated[0][1]) > self.mem_limit:
                break

            othertopref, othersize = cur_activated.pop(0)
            new_activated.append((othertopref, othersize))
            total += othersize

        log_str = f"Activating {topref().__class__.__name__}"
        if self.name:
            log_str += f" on {self.name}"
        if cur_activated:
            log_str += f", removing {', '.join([x[0]().__class__.__name__ for x in cur_activated])}"
        logger.debug(log_str)

        logger.debug(
            f"New set {len(new_activated)}, {total//1024//1024}MB, of {self.max_activated}, {self.mem_limit // 1024 // 1024}MB max"
        )

        self.activated = new_activated

        self.reset(exclude=[x[0] for x in self.activated])


def clone_model(
    model,
    clone_tensors: Literal["share"] | str | torch.device = "share",
    exclusion_set=None,
):
    """
    Copies a model so you get a different set of instances, but they share
    all their parameters and buffers
    """

    # If this isn't actually a model, just return a deepcopy
    if not isinstance(model, torch.nn.Module):
        clone = deepcopy(model)
        if clone_tensors != "share" and hasattr(clone, "to"):
            clone = clone.to(clone_tensors)
        return clone

    # Start by pulling all the Tensors out of the model, so they're not copied on deepclone
    cache = {}

    for (model_name, source) in model.named_modules():
        model_params = {}
        model_buffers = {}

        for name, param in source.named_parameters(recurse=False):
            model_params[name] = param
            source._parameters[name] = None

        for name, buffer in source.named_buffers(recurse=False):
            model_buffers[name] = buffer
            source._buffers[name] = None

        cache[model_name] = (model_params, model_buffers)

    # Deep clone the model
    clone = deepcopy(model)

    # Put the tensors back into the model
    for (model_name, dest) in model.named_modules():
        model_params, model_buffers = cache[model_name]

        for name, param in model_params.items():
            dest._parameters[name] = param
        for name, buffer in model_buffers.items():
            dest._buffers[name] = buffer

    # And into the clone
    # Even if we're not sharing, set it to shared to start with
    for (model_name, dest) in clone.named_modules():
        model_params, model_buffers = cache[model_name]

        for name, param in model_params.items():
            dest.register_parameter(name, param)
        for name, buffer in model_buffers.items():
            dest.register_buffer(name, buffer)

    if clone_tensors != "share":
        for (model_name, dest) in clone.named_modules():
            model_params, model_buffers = cache[model_name]

            if exclusion_set:
                for name in model_params.keys():
                    set_module_tensor_to_device(dest, name, "meta")
                for name in model_buffers.keys():
                    set_module_tensor_to_device(dest, name, "meta")

                add_hook(
                    dest,
                    CloneToGPUHook(
                        clone_tensors,
                        exclusion_set.activate_for(clone),
                        model_params,
                        model_buffers,
                    ),
                    replace=True,
                )
            else:
                for name, param in model_params.items():
                    new_param = param.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_param)
                for name, buffer in model_buffers.items():
                    new_buffer = buffer.to(clone_tensors, copy=True)
                    set_module_tensor_to_device(dest, name, clone_tensors, new_buffer)

        if exclusion_set:
            exclusion_set.add(clone)

    return clone


class FILOSequentialHook(SequentialHook):
    """
    SequentialHook executes pre_forward and post_forward in the same order.
    Bu usually we want post_forward hooks to be called backwards (so the last
    hook to have pre_forward called is the first to have post_forward called)
    """

    def post_forward(self, module, output):
        for hook in reversed(self.hooks):
            output = hook.post_forward(module, output)
        return output


def is_hooked(module):
    return hasattr(module, "_hf_hook")


def get_hooks(module, hook_class):
    if hasattr(module, "_hf_hook"):
        if isinstance(module._hf_hook, SequentialHook):
            for hook in module._hf_hook.hooks:
                if isinstance(hook, hook_class):
                    yield hook

        else:
            if isinstance(module._hf_hook, hook_class):
                yield module._hf_hook


def get_hook(module, hook_class, error_if_multiple=True):
    hooks = list(get_hooks(module, hook_class))

    if hooks:
        if len(hooks) > 1 and error_if_multiple:
            raise RuntimeError(f"Found more than one {hook_class.__name__}")
        return hooks[0]

    return False


def has_hook(module, hook_class):
    return any(get_hooks(module, hook_class))


def add_hook(module, hook, replace=False):
    append = not replace

    if append and (getattr(module, "_hf_hook", None) is not None):
        if isinstance(module._hf_hook, SequentialHook):
            if not isinstance(module._hf_hook, FILOSequentialHook):
                logger.warn("Non-FILO Sequential Hook found")

            module = hook.init_hook(module)
            module._hf_hook.hooks = (*module._hf_hook.hooks, hook)
            return

        else:
            old_hook = module._hf_hook
            remove_hook_from_module(module)
            hook = FILOSequentialHook(old_hook, hook)

    add_hook_to_module(module, hook, append=False)


def remove_hook(module, hook_class):
    if hasattr(module, "_hf_hook"):
        if isinstance(module._hf_hook, SequentialHook):
            for removed in [
                hook
                for hook in module._hf_hook.hooks
                if not isinstance(hook, hook_class)
            ]:
                removed.detach_hook(module)

            module._hf_hook.hooks = [
                hook
                for hook in module._hf_hook.hooks
                if not isinstance(hook, hook_class)
            ]
            if not module._hf_hook.hooks:
                remove_hook_from_module(module)

        elif isinstance(module._hf_hook, hook_class):
            remove_hook_from_module(module)


def replace_hooked_forward(module, forward):
    old_forward = module._old_forward

    @functools.wraps(old_forward)
    def new_forward(*args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        if module._hf_hook.no_grad:
            with torch.no_grad():
                output = forward(*args, **kwargs)
        else:
            output = forward(*args, **kwargs)
        return module._hf_hook.post_forward(module, output)

    module.forward = new_forward


def restore_hooked_forward(module):
    replace_hooked_forward(module, module._old_forward)
