from typing import Literal

import kornia
import torch
from diffusers.models import modeling_utils

from gyre import images

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


class InSPyReNetPipeline:
    def __init__(self, module):
        self.module = module

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [("module", self.module)]

    @torch.no_grad()
    def __call__(
        self, tensor, mode: Literal["alpha", "solid", "mask"] = "alpha", color=[0, 0, 0]
    ):
        """
        Color must be 0..1 if used, not 0..255
        """
        if tensor.ndim != 4 or tensor.shape[2] < 3:
            raise ValueError("Tensor must be RGB image in BCHW format")

        tensor = tensor[:, [0, 1, 2]]  # Discard any existing alpha

        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)

        sample = kornia.enhance.normalize(tensor, MEAN, STD)
        sample = sample.to(device, dtype)

        pred = self.module(sample)
        pred = pred.to(tensor.device, tensor.dtype)

        if mode == "mask":
            return pred

        elif mode == "alpha":
            return torch.cat([tensor, pred], dim=1)

        elif mode == "solid":
            colort = torch.tensor(color).to(tensor.device, tensor.dtype)
            overlay = (torch.ones_like(tensor).T * colort.unsqueeze(1)).T
            return tensor * pred + overlay * (1 - pred)

        else:
            raise ValueError(f"Unknown background removal mode {mode}")

        print(tensor.shape, pred.shape)
