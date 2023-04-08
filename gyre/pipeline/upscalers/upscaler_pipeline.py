import logging

import torch
from diffusers.models import modeling_utils
from diffusers.pipeline_utils import DiffusionPipeline
from PIL.Image import Image as PILImage

from gyre import images
from gyre.pipeline.prompt_types import ImageLike
from gyre.pipeline.upscalers.utils import tile

logger = logging.getLogger(__name__)


class UpscalerPipeline(DiffusionPipeline):

    # Configuration for DiffusionPipelineWrapper
    _meta = {"offload_capable": False}

    module: torch.nn.Module

    def __init__(self, module):
        super().__init__()

        self.register_modules(module=module)

    def to(self, device):
        self.module.to(device)

    @torch.no_grad()
    def __call__(
        self,
        image: ImageLike | None = None,
        width: int | None = None,
        height: int | None = None,
        **kwargs,
    ):

        # Handle PIL Image
        if isinstance(image, PILImage):
            image = images.fromPIL(image)

        if height == image.shape[-2] and width == image.shape[-1]:
            logger.debug("Ignoring passed size, since it matches input image")
            height = width = None

        device = modeling_utils.get_parameter_device(self.module)
        dtype = modeling_utils.get_parameter_dtype(self.module)

        sample = image.to(device, dtype)
        alpha = None

        if sample.shape[1] == 4:
            alpha = sample[:, [3]]
            sample = sample[:, 0:3]

        scale = getattr(self.module, "_scale", 4)
        window_size = getattr(self.module, "_window_size", 32)

        result = tile(sample[:, :3], self.module, scale, 256).clamp(0, 1)

        if alpha is not None:
            # If alpha is a single unique value, just replicate rather than scaling
            if len(unique := alpha.unique_consecutive()) == 1:
                outshape = images.scale_shape(alpha.shape, scale)
                result_alpha = torch.ones(outshape).to(result) * unique[0]

            else:
                result_alpha = images.resize(alpha, scale)

            result = torch.cat([result, result_alpha], dim=1)

        if width is not None and height is not None:
            if width > result.shape[-1] or height > result.shape[-2]:
                logger.warn(
                    f"Requested size ({width}x{height}) is greated than result after upscale ({result.shape[-2]}x{result.shape[-1]}). "
                    "Further lower-quality upscaling will occur."
                )

            result = images.rescale(result, height, width, "cover")

        return (result.to(image),)
