import torch.nn as nn
from accelerate.hooks import ModelHook
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from gyre.pipeline.model_utils import add_hook, has_hook, remove_hook

"""
Includes patchers to adjust a unet to support applying a T2I-adapter adapter_states
to CrossAttnDownBlock2D and UNet2DConditionModel

Probably pretty brittle - the places adapter_states need handling are right in
the middle of quite long functions. These patches depend on the specific structure
of those methods.

(In the long run, just patching in the whole modified class via module cache pollution 
might be a better strategy)
"""


class DownsamplerWrapper(nn.Module):
    def __init__(self, downsampler, adapter_state):
        super().__init__()
        self.downsampler = downsampler
        self.adapter_state = adapter_state

    def forward(self, hidden_states):
        hidden_states += self.adapter_state
        return self.downsampler(hidden_states)


class CrossAttnDownBlock2DHook(ModelHook):
    def pre_forward(self, module, *args, **kwargs):
        self.adapter_state = kwargs.pop("adapter_state", None)

        if self.adapter_state is not None:
            if module.downsamplers is not None:
                # If downsamplers is not None, wrap the first downsampler to add
                # adapter_state to hidden_state just before that first downsampler call
                module.downsamplers._modules["0"] = DownsamplerWrapper(
                    module.downsamplers._modules["0"], self.adapter_state
                )

        return args, kwargs

    def post_forward(self, module, output):
        hidden_states, output_states = output

        if self.adapter_state is not None:
            if module.downsamplers is not None:
                # Restore the downsampler to it's unwrapped state
                original_downsampler = module.downsamplers._modules["0"].downsampler
                module.downsamplers._modules["0"] = original_downsampler
            else:
                # Otherwise, adapter_state wasn't added just prior to the downsampler
                # call, so do it now
                hidden_states += self.adapter_state

        return hidden_states, output_states


class DownBlockWrapper(nn.Module):
    """
    Wraps a down block to add in an adapter_state, either as an argument to the block
    (in the case of a Cross Attention block) or directly.

    Behaviour is very dependant on the specific internals of UNet2DConditionModel.foward
    """

    def __init__(self, down_block, adapter_state):
        super().__init__()
        self.down_block = down_block
        self.adapter_state = adapter_state
        self.has_cross_attention = getattr(
            self.down_block, "has_cross_attention", False
        )

    def forward(self, *args, **kwargs):
        if self.has_cross_attention:
            return self.down_block(*args, **kwargs, adapter_state=self.adapter_state)
        else:
            sample, res_sample = self.down_block(*args, **kwargs)
            sample += self.adapter_state
            return sample, res_sample


class UNet2DConditionModelHook(ModelHook):
    """
    A hook to apply to a UNet2DConditionModel to wrap the down blocks in a
    DownBlockWrapper just before forward is called, and remove them afterwards
    """

    def __init__(self):
        super().__init__()
        self.old_blocks = None

    def pre_forward(self, module, *args, **kwargs):
        adapter_states = kwargs.pop("adapter_states", None)
        self.old_blocks = None

        if adapter_states:
            assert len(module.down_blocks) == len(adapter_states)

            self.old_blocks, module.down_blocks = module.down_blocks, nn.ModuleList(
                [
                    DownBlockWrapper(down_block, adapter_state)
                    for down_block, adapter_state in zip(
                        module.down_blocks, adapter_states
                    )
                ]
            )

        return args, kwargs

    def post_forward(self, module, output):
        if self.old_blocks:
            module.down_blocks = self.old_blocks

        return output


# ------------------------


def unpatch_unet(unet):
    for module in unet.modules():
        if isinstance(module, CrossAttnDownBlock2D):
            remove_hook(module, CrossAttnDownBlock2DHook)
        elif isinstance(module, UNet2DConditionModel):
            remove_hook(module, UNet2DConditionModelHook)


def patch_cross_attn_down_block_2D(module: CrossAttnDownBlock2D):
    if not has_hook(module, CrossAttnDownBlock2DHook):
        add_hook(module, CrossAttnDownBlock2DHook())


def patch_unet_2D_condition_model(module: UNet2DConditionModel):
    if not has_hook(module, UNet2DConditionModelHook):
        add_hook(module, UNet2DConditionModelHook())


def patch_unet(unet):
    for module in unet.modules():
        if isinstance(module, CrossAttnDownBlock2D):
            patch_cross_attn_down_block_2D(module)
        elif isinstance(module, UNet2DConditionModel):
            patch_unet_2D_condition_model(module)
