from typing import Literal

import kornia
import torch
import torchvision
from diffusers.models import modeling_utils

from gyre import images

IMAGENET_MEAN = [0.485, 0.456, 0.406]


# TODO: This is basically the same as informative_drawing_pipeline
class HedPipeline:
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

        # ImageNet mean
        offset = torch.tensor(IMAGENET_MEAN).to(sample)
        offset = offset.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(sample)

        # Prepare sample
        sample = sample - offset
        sample = sample[:, [2, 1, 0]] * 255  # Convert to BGR, 0..255

        result = images.normalise_range(self.module(sample)[-1])

        return result.to(tensor.device, tensor.dtype)
