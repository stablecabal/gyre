from typing import Literal

import kornia
import torch
import torchvision
from diffusers.models import modeling_utils

from gyre import images


class BaenormalPipeline:
    module: torch.nn.Module

    def __init__(self, module):
        self.module = module
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [("module", self.module), ("normalize", self.normalize)]

    @torch.no_grad()
    def __call__(self, tensor):
        tensor = images.normalise_tensor(tensor, 3)

        sample = tensor

        # Get device and dtype of model
        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)
        sample = sample.to(device, dtype)

        sample = self.normalize(sample)
        norm_out_list, _, _ = self.module(sample)
        norm_out = norm_out_list[-1][:, :3]

        # Normalise the vector length to a unit vector
        veclen = (norm_out**2.0).sum(dim=1) ** 0.5
        norm_out /= veclen

        print(norm_out.min(), norm_out.max())

        # Convert from -1..1 to 0..1
        result = (norm_out * 0.5 + 0.5).clip(0, 1)

        return result.to(tensor.device, tensor.dtype)
