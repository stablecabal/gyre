from typing import Literal

import torch
from diffusers.models import modeling_utils
from mmseg.apis import inference_segmentor
from mmseg.core.evaluation import get_palette

from gyre import images


class MmsegPipeline:
    def __init__(self, module):
        self.module = module

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [("module", self.module)]

    @torch.no_grad()
    def __call__(self, tensor):
        if tensor.ndim != 4 or tensor.shape[2] < 3:
            raise ValueError("Tensor must be RGB image in BCHW format")

        samples = [image for image in images.toCV(tensor[:, [0, 1, 2]])]
        results = inference_segmentor(self.module, samples)

        outputs = []

        for sample, result in zip(samples, results):

            # module = self.module
            # if hasattr(module, "module"):
            #     print("!")
            #     module = module.module

            output = self.module.show_result(
                sample, [result], palette=get_palette("ade"), show=False, opacity=1
            )

            outputs.append(images.fromCV(output))

        return torch.cat(outputs, dim=0).to(tensor.device, tensor.dtype)
