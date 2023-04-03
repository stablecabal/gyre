import math
from typing import Literal

import kornia
import torch
from diffusers.models import modeling_utils

from gyre import images
from gyre.pipeline.hinters.models.guided_filter import guidedfilter2d_color

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
MAX_RES = 1280


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
        scale = None
        padding = None
        guided_filter = None

        if tensor.shape[-1] > MAX_RES or tensor.shape[-2] > MAX_RES:
            # Pad up to square
            maxdim = max(tensor.shape[-2], tensor.shape[-1])

            padding = (maxdim - tensor.shape[-1], 0, maxdim - tensor.shape[-2], 0)
            sample = torch.nn.functional.pad(tensor, padding, mode="reflect")

            # Scale down to 1280 x 1280
            scale = min(MAX_RES / tensor.shape[-2], MAX_RES / tensor.shape[-1])

            sample = images.resize(sample, scale).contiguous()

            # Use guided filter if tensor is over double MAX_RES
            guided_filter = scale < 0.5

        else:
            # Pad up to square, plus whatever to get to multiple of 32
            maxdim = max(
                math.ceil(tensor.shape[-1] / 32) * 32,
                math.ceil(tensor.shape[-2] / 32) * 32,
            )

            padding = (maxdim - tensor.shape[-1], 0, maxdim - tensor.shape[-2], 0)
            sample = torch.nn.functional.pad(tensor, padding, mode="reflect")

        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)

        sample = kornia.enhance.normalize(sample, MEAN, STD)
        sample = sample.to(device, dtype)

        pred = self.module(sample)
        pred = pred.to(tensor.device, tensor.dtype)

        if scale is not None:
            pred = images.resize(pred, 1 / scale)

        if padding is not None:
            pred = pred[:, :, padding[2] :, padding[0] :]

        if guided_filter:
            print("Guided Filtering Mask")

            guided_pred = (
                guidedfilter2d_color(
                    tensor.to(torch.float64), pred.to(torch.float64), 32, 1e-8
                )
                .clamp(0, 1)
                .to(tensor.dtype)
            )

            pred = torch.max(guided_pred, pred)

        # Slightly shrink mask
        pred = ((pred - 0.2) / 0.8).clamp(0, 1)

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
