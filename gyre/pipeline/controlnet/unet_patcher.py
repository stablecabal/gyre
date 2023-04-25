import inspect
import logging

import torch.nn as nn
from accelerate.hooks import ModelHook
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from gyre.pipeline.model_utils import add_hook, has_hook, remove_hook

logger = logging.getLogger(__name__)

"""
Includes patchers to adjust a unet to support applying ControlNet residuals
"""


class UpBlockWrapper(nn.Module):
    """
    Wraps an upblock to add an additional residual to the passed in
    res_state
    """

    def __init__(self, up_block, residuals):
        super().__init__()
        self.up_block = up_block
        self.residuals = residuals
        self.has_cross_attention = getattr(self.up_block, "has_cross_attention", False)

    @property
    def resnets(self):
        return self.up_block.resnets

    def forward(self, *args, **kwargs):
        res_samples = kwargs.pop("res_hidden_states_tuple")

        res_samples = tuple(
            (
                res_sample + extra
                for res_sample, extra in zip(res_samples, self.residuals)
            )
        )

        return self.up_block(*args, **kwargs, res_hidden_states_tuple=res_samples)


class MidBlockWrapper(nn.Module):
    def __init__(self, mid_block, residual):
        super().__init__()
        self.mid_block = mid_block
        self.residual = residual

    def forward(self, *args, **kwargs):
        return self.mid_block(*args, **kwargs) + self.residual


class UNet2DConditionModelHook(ModelHook):
    """
    A hook to apply to a UNet2DConditionModel to wrap the down blocks in a
    DownBlockWrapper just before forward is called, and remove them afterwards
    """

    def __init__(self):
        super().__init__()
        self.old_mid_block = None
        self.old_up_blocks = None

    def pre_forward(self, module, *args, **kwargs):
        down_residuals = kwargs.pop("down_block_additional_residuals", None)
        mid_residual = kwargs.pop("mid_block_additional_residual", None)

        self.old_mid_block = None
        self.old_up_blocks = None

        if down_residuals is not None:
            residual_chunks = []
            for block in module.up_blocks:
                chunk_size = len(block.resnets)
                residual_chunks.append(down_residuals[-chunk_size:])
                down_residuals = down_residuals[:-chunk_size]

            (self.old_up_blocks, module.up_blocks,) = module.up_blocks, nn.ModuleList(
                [
                    UpBlockWrapper(up_block, res_chunk)
                    for up_block, res_chunk in zip(module.up_blocks, residual_chunks)
                ]
            )

        if mid_residual is not None:
            self.old_mid_block, module.mid_block = module.mid_block, MidBlockWrapper(
                module.mid_block, mid_residual
            )

        return args, kwargs

    def post_forward(self, module, output):
        if self.old_up_blocks:
            module.up_blocks = self.old_up_blocks
        if self.old_mid_block:
            module.mid_block = self.old_mid_block

        return output


# ------------------------


def unpatch_unet(unet):
    for module in unet.modules():
        if isinstance(module, UNet2DConditionModel):
            remove_hook(module, UNet2DConditionModelHook)


def patch_unet_2D_condition_model(module: UNet2DConditionModel):
    if not has_hook(module, UNet2DConditionModelHook):
        add_hook(module, UNet2DConditionModelHook())


def patch_unet(unet):
    for module in unet.modules():
        if isinstance(module, UNet2DConditionModel):

            # If any of the unet modules already take down_block_additional_residuals
            # this is Diffusers >= 0.14.0 and we don't need to patch

            # TODO: Disbled as the built-in support breaks CLIP guidance
            if False:
                unet_keys = inspect.signature(module.forward).parameters.keys()
                if "down_block_additional_residuals" in unet_keys:
                    logger.info("Unet2DConditionModel already suppports ControlNet")
                    return

            patch_unet_2D_condition_model(module)
