from math import floor

import kornia
import numpy as np
import torch
import tqdm
from diffusers.models import modeling_utils
from diffusers.pipeline_utils import DiffusionPipeline
from PIL.Image import Image as PILImage

from gyre import images
from gyre.pipeline.prompt_types import ImageLike


def tile(
    input,
    module,
    scale,
    tile_size,
    tile_overlap=0.25,
    window_size=32,
    pad_size=None,
    pad_mode="reflect",
    feather=None,
):
    # Check ndim
    assert input.ndim == 4, "Tile input must be in BCHW format"

    # Check channels
    assert input.shape[1] == 3, "Tile input must be RGB"

    # Check tile_size
    assert (
        tile_size % window_size == 0
    ), f"Tile size {tile_size} needs to be multiple of {window_size}"

    # Get input shape
    ih, iw = input.shape[-2], input.shape[-1]

    # Make sure tile size is <= input size (while still being divisible by window_size)
    tile_size = min(
        tile_size,
        floor(ih / window_size) * window_size,
        floor(iw / window_size) * window_size,
    )

    # Pad input
    pad_size = window_size if pad_size is None else pad_size

    pad_tuple = (pad_size, pad_size, pad_size, pad_size)
    padded_input = torch.nn.functional.pad(input, pad_tuple, pad_mode)
    pih, piw = padded_input.shape[-2], padded_input.shape[-1]

    # Create inital output:
    output = torch.zeros(images.scale_shape(padded_input.shape, scale)).to(input)
    out_tile = tile_size * scale

    # TODO Probably a way to calculcate this directly

    # Iterively increase tile_count until overlap is met
    tiles_y = 1 if tile_size >= pih else 2
    tiles_x = 1 if tile_size >= piw else 2

    while tiles_y > 1:
        overlap_y = ((tile_size * tiles_y) - pih) / (tiles_y - 1)

        if overlap_y < tile_overlap * tile_size:
            tiles_y += 1
        else:
            break

    while tiles_x > 1:
        overlap_x = ((tile_size * tiles_x) - piw) / (tiles_x - 1)

        if overlap_x < tile_overlap * tile_size:
            tiles_x += 1
        else:
            break

    # Calculate the feather map,

    feather = pad_size if feather is None else feather

    assert tile_overlap * tile_size > feather, "Tile overlap must exceed feather size"

    # Feather is on output, not input
    feather *= scale
    # Gaussian blur means we need to fill white to the center of the feather
    feather //= 2

    feather_shape = [input.shape[0], 1, out_tile - feather * 2, out_tile - feather * 2]

    feather_map = torch.ones(feather_shape).to(input)
    feather_map = torch.nn.functional.pad(feather_map, (feather,) * 4, "constant", 0)
    feather_map = images.gaussianblur(feather_map, feather / 3)

    y_offsets = np.linspace(0, pih - tile_size, tiles_y).round().astype(np.uint32)
    x_offsets = np.linspace(0, piw - tile_size, tiles_x).round().astype(np.uint32)

    with tqdm.tqdm(total=tiles_x * tiles_y) as t:

        for y in y_offsets:
            oy = y * scale
            for x in x_offsets:
                ox = x * scale

                existing = output[:, :, oy : oy + out_tile, ox : ox + out_tile]
                new = module(padded_input[:, :, y : y + tile_size, x : x + tile_size])
                mixed = new * feather_map + existing * (1 - feather_map)

                output[:, :, oy : oy + out_tile, ox : ox + out_tile] = mixed
                t.update()

    return output[
        :,
        :,
        pad_size * scale : pad_size * scale + ih * scale,
        pad_size * scale : pad_size * scale + iw * scale,
    ]


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
            print("Ignoring passed size, since it matches input image")
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
            factors = (height / result.shape[-2], width / result.shape[-1])
            result = images.resize(result, factors)

        return (result.to(image),)
