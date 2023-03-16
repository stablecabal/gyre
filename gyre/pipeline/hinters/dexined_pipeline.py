from typing import Literal

import kornia
import torch
from diffusers.models import modeling_utils


class DexinedPipeline:
    module: torch.nn.Module

    def __init__(self, module):
        self.module = module

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [("module", self.module)]

    def _preprocess(self, tensor):
        tensor = tensor[:, [0, 1, 2]]
        return tensor * 255.0

    def _execute(self, tensor):
        return self.module(tensor)[-1]

    def _postprocess(self, tensor):
        # Normalise
        r_min = torch.amin(tensor, dim=[1, 2, 3], keepdim=True)
        r_max = torch.amax(tensor, dim=[1, 2, 3], keepdim=True)

        # tensor = (tensor - r_min) / (r_max - r_min)
        tensor = tensor.clamp(0, 1)
        # tensor = kornia.enhance.adjust_gamma(tensor, 5.0)
        # tensor = torch.nn.functional.threshold(tensor, 0.2, 0)

        return tensor

    @torch.no_grad()
    def __call__(self, tensor):
        if tensor.ndim != 4 or tensor.shape[2] < 3:
            raise ValueError("Tensor must be RGB image in BCHW format")

        sample = tensor

        # Get device and dtype of model
        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)
        sample = sample.to(device, dtype)

        sample = self._preprocess(sample)
        sample = self._execute(sample)
        sample = self._postprocess(sample)

        return sample.to(tensor.device, tensor.dtype)
