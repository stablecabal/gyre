from typing import Literal

import kornia
import torch
import torchvision
from diffusers.models import modeling_utils

from gyre import images


class InformativeDrawingPipeline:
    module: torch.nn.Module

    def __init__(self, module):
        self.module = module

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [("module", self.module)]

    @torch.no_grad()
    def __call__(self, tensor):
        tensor = images.normalise_tensor(tensor, 3)

        sample = tensor

        # Get device and dtype of model
        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)
        sample = sample.to(device, dtype)

        result = self.module(sample).clamp(0, 1)

        # Change to white on a black background
        # TODO - have ControlNet understand autoinvert the same way T2I does
        result = 1 - result

        return result.to(tensor.device, tensor.dtype)
