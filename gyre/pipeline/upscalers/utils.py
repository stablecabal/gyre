from math import floor

import numpy as np
import torch
import tqdm

from gyre import images


def tile(
    input,
    module,
    scale: float,
    tile_size: int,
    tile_overlap=0.25,
    window_size: int = 32,
    pad_size: int | None = None,
    pad_mode="reflect",
    feather: int | None = None,
    progress_bar=tqdm.tqdm,
    **progress_bar_kwargs,
):
    # Check ndim
    assert input.ndim == 4, "Tile input must be in BCHW format"

    # Check tile_size
    assert (
        tile_size % window_size == 0
    ), f"Tile size {tile_size} needs to be multiple of {window_size}"

    # Get input shape
    ih, iw = input.shape[-2], input.shape[-1]

    # If tile_size is at least image height & width, no need to tile
    if tile_size >= ih and tile_size >= iw:
        return module(input)

    # Othewise make sure tile size is <= input size, while still being divisible by window_size
    # (one could be greater while the other is smaller)
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

    # Find output size
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

    assert pad_size >= feather, "Pad size must equal or exceed feather size"
    assert (
        tile_overlap * tile_size >= feather
    ), f"Tile overlap ({tile_overlap * tile_size}) must equal or exceed feather size {feather}"

    if feather:
        # Feather is on output, not input
        feather = round(feather * scale)
        # Gaussian blur means we need to fill white to the center of the feather
        feather //= 2

        feather_shape = [1, 1, out_tile - feather * 2, out_tile - feather * 2]

        feather_map = torch.ones(feather_shape).to(input)
        feather_map = torch.nn.functional.pad(
            feather_map, (feather,) * 4, "constant", 0
        )
        feather_map = images.gaussianblur(feather_map, feather / 3)
    else:
        feather_map = None

    y_offsets = np.linspace(0, pih - tile_size, tiles_y).round().astype(np.uint32)
    x_offsets = np.linspace(0, piw - tile_size, tiles_x).round().astype(np.uint32)

    # We don't know the output batch size or channel count yet, so lazily instantiate when we do
    output = None

    def check_output(t):
        nonlocal output

        if output is None:
            output = torch.zeros(
                t.shape[0],
                t.shape[1],
                *images.scale_shape(padded_input.shape, scale)[-2:],
            ).to(t)
        else:
            assert output.shape[0] == t.shape[0] and output.shape[1] == t.shape[1]

    with progress_bar(total=tiles_x * tiles_y, **progress_bar_kwargs) as t:

        for y in y_offsets:
            oy = y * scale
            for x in x_offsets:
                ox = x * scale

                new = module(padded_input[:, :, y : y + tile_size, x : x + tile_size])
                check_output(new)
                existing = output[:, :, oy : oy + out_tile, ox : ox + out_tile]

                if feather_map is None:
                    mixed = new
                else:
                    mixed = new * feather_map + existing * (1 - feather_map)

                output[:, :, oy : oy + out_tile, ox : ox + out_tile] = mixed
                t.update()

    return output[
        :,
        :,
        pad_size * scale : pad_size * scale + ih * scale,
        pad_size * scale : pad_size * scale + iw * scale,
    ]
