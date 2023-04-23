# Utility functions for handling images as PyTorch Tensors

# All images in are in BCHW unless specified in the variable name, as floating point 0..1
# All functions will handle RGB or RGBA images

import math
import struct
import typing
import zlib
from math import ceil, sqrt
from typing import Literal

import cv2 as cv
import kornia
import numpy as np
import torch
import torchvision
from PIL import Image as PILImage
from tqdm import tqdm, trange

from gyre.images_shuffle import ContentShuffleDetector
from gyre.match_histograms import match_histograms as np_match_histograms

from .resize_right import interp_methods
from .resize_right import resize as resize_right


def fromPIL(image):
    # Get as numpy HWC 0..1
    rgbHWC = np.array(image).astype(np.float32) / 255.0
    # Convert to BCHW
    rgbBCHW = rgbHWC[None].transpose(0, 3, 1, 2)
    # And convert to Tensor
    return torch.from_numpy(rgbBCHW)


def toPIL(tensor):
    # Convert to BCHW if just CHW
    if tensor.ndim == 3:
        tensor = tensor[None, ...]
    # Then convert to BHWC
    rgbBHWC = tensor.permute(0, 2, 3, 1)
    # Then convert from 0..1 to 0..255
    images = (rgbBHWC.to(torch.float32) * 255).round().to(torch.uint8).cpu().numpy()
    # And put into PIL image instances
    return [PILImage.fromarray(image) for image in images]


def fromCV(bgrHWC):
    tensor = torch.from_numpy(bgrHWC).to(torch.float32) / 255.0

    # Handle mono (no channels at all) - add one channel to the right
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(dim=-1)
    # Expand to BHWC
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(dim=0)
    # Convert to BCHW
    bgrBCHW = tensor.permute(0, 3, 1, 2)
    # Convert from BGR to RGB (if there are enough channels)
    if bgrBCHW.shape[1] >= 3:
        channels = [2, 1, 0, 3][: bgrBCHW.shape[1]]
        bgrBCHW = bgrBCHW[:, channels]
    # Done
    return bgrBCHW


def toCV(tensor):
    if tensor.ndim == 3:
        tensor = tensor[None, ...]

    # Handle single channel
    if tensor.shape[1] < 3:
        bgrBCHW = tensor
    # Handle 3 / 4 channel, convert rgb(a) to bgr(a)
    else:
        bgrBCHW = tensor[:, [2, 1, 0, 3][: tensor.shape[1]]]

    # Move from channel-last to channel-first
    bgrBHWC = bgrBCHW.permute(0, 2, 3, 1)

    return (bgrBHWC.to(torch.float32) * 255).round().to(torch.uint8).cpu().numpy()


def fromPngBytes(bytes):
    intensor = torch.tensor(np.frombuffer(bytes, dtype=np.uint8))
    asuint8 = torchvision.io.decode_image(
        intensor, torchvision.io.image.ImageReadMode.RGB_ALPHA
    )
    return asuint8[None, ...].to(torch.float32) / 255


def toPngBytes(tensor):
    tensor = tensor.to("cpu")

    if tensor.ndim == 3:
        tensor = tensor[None, ...]

    if tensor.shape[1] == 1 or tensor.shape[1] == 3:
        tensor = (tensor.to(torch.float32) * 255).round().to(torch.uint8)
        pngs = [torchvision.io.encode_png(image) for image in tensor]
        return [png.numpy().tobytes() for png in pngs]
    elif tensor.shape[1] == 4:
        images = toCV(tensor)
        return [cv.imencode(".png", image)[1].tobytes() for image in images]
    else:
        print(f"Don't know how to save PNGs with {tensor.shape[1]} channels")
        return []


SUPPORTS_WEBP = any(
    (
        "build" in line or "yes" in line
        for line in cv.getBuildInformation().lower().splitlines()
        if "webp:" in line
    )
)


def fromWebpBytes(bytes):
    cvimage = cv.imdecode(np.frombuffer(bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    return normalise_tensor(fromCV(cvimage), 4)


def toWebpBytes(tensor):
    if tensor.ndim == 3:
        tensor = tensor[None, ...]

    images = toCV(tensor)

    # quality > 100 == lossless
    return [
        cv.imencode(".webp", image, [cv.IMWRITE_WEBP_QUALITY, 500])[1].tobytes()
        for image in images
    ]


def fromAuto(image):
    if isinstance(image, torch.Tensor):
        return normalise_tensor(image, channels=None)

    # Handle PIL image
    elif isinstance(image, PILImage.Image):
        return fromPIL(image)

    # Handle nparray in various forms (0..1, 0..255, BCHW / BHWC / CHW / HWC format)
    elif isinstance(image, np.ndarray):
        # Handle 2 dim (mono, HW) or 3 dim (CHW or HWC)
        while image.ndim < 4:
            image = image[None, ...]

        # Determine if BCHW or BHWC by assuming H and W will be > 16, and C will be <
        if image.shape[1] < 16:
            image = np.moveaxis(image, 1, 3)

        # Detect if range is 0..1 or 0..255 already
        if image.dtype != np.uint8 and image.max() <= 1:
            image = (image * 255).round().astype("uint8")

        return fromCV(image)

    raise RuntimeError("Can't determine type of image to convert to tensor")


def addTextChunkToPngBytes(binary, key: str, text: str):
    txt_chunk = key.encode("utf-8") + b"\0" + text.encode("utf-8")

    # check for last chunk
    iend_index = binary.rindex(b"IEND") - 4
    img_data_pre = binary[:iend_index]
    img_data_post = binary[iend_index:]

    # new chunk
    length = struct.pack(">I", len(txt_chunk))
    chunktype = b"tEXt"
    chunk_crc = zlib.crc32(chunktype)
    chunk_crc = zlib.crc32(txt_chunk, chunk_crc)
    chunk_crc = struct.pack(">I", chunk_crc)

    # return new binary with tEXt chunk injected in the middle
    return b"".join(
        (img_data_pre, length, chunktype, txt_chunk, chunk_crc, img_data_post)
    )


def addTextChunkToWebpBytes(binary, info_fourcc: bytes, text: str):
    assert len(info_fourcc) == 4, f"{info_fourcc} needs to a be four byte FourCC"

    txt_body = text.encode("utf-8")
    txt_chunk = (
        b"LIST"
        + struct.pack("<I", 4 + 4 + 4 + len(txt_body))
        + b"INFO"
        + info_fourcc
        + struct.pack("<I", len(txt_body))
        + txt_body
        + bytes(len(txt_body) % 2)
    )

    # Check if this is simple or complex webp
    if binary[12:16] == b"VP8L":
        remainder = binary[12:]
        info = struct.unpack("<I", binary[21:25])[0]
        width = (info & 0x3FFF) + 1
        height = ((info >> 14) & 0x3FFF) + 1
        alpha = bool((info >> 28) & 0x1)

        header_chunk = (
            b"VP8X"
            + struct.pack("<I", 10)
            + bytes([0b10000 if alpha else 0, 0, 0, 0])
            + struct.pack("<I", width - 1)[0:3]
            + struct.pack("<I", height - 1)[0:3]
        )

        totallen = 4 + len(header_chunk) + len(remainder) + len(txt_chunk)

        return (
            b"RIFF"
            + struct.pack("<I", totallen)
            + b"WEBP"
            + header_chunk
            + remainder
            + txt_chunk
        )

    raise NotImplementedError("Can only apply chunk to simple lossless webp")


def normalise_tensor(tensor: torch.Tensor, channels: int | None = 3) -> torch.Tensor:
    # Process depth map
    if tensor.ndim == 3:
        tensor = tensor[None, ...]

    if channels is None:
        return tensor
    elif channels == 1:
        return tensor[:, [0]]
    elif channels == 3:
        return tensor[:, [0, 1, 2]] if tensor.shape[1] >= 3 else tensor[:, [0, 0, 0]]
    elif channels == 4:
        if tensor.shape[1] >= 4:
            return tensor[:, [0, 1, 2, 3]]
        else:
            alpha = torch.ones(tensor.shape[0], 1, *tensor.shape[2:]).to(tensor)
            tensor = normalise_tensor(tensor, 3)
            return torch.concat((tensor, alpha), dim=1)

    raise ValueError(f"Unknown number of channels {channels}")


def normalise_range(tensor: torch.Tensor) -> torch.Tensor:
    dmin = torch.amin(tensor, dim=[1, 2, 3], keepdim=True)
    dmax = torch.amax(tensor, dim=[1, 2, 3], keepdim=True)
    return (tensor - dmin) / (dmax - dmin)


# TOOD: This won't work on images with alpha
def levels(tensor, in0, in1, out0, out1):
    c = (out1 - out0) / (in1 - in0)
    return ((tensor - in0) * c + out0).clamp(0, 1)


def invert(tensor):
    return 1 - tensor


# 0, 1, 2, 3 = r, g, b, a | 4 = 0 | 5 = 1 | 6 = drop
# TODO: These are from generation.proto, but we should be nicer about the mapping
def channelmap(tensor, srcchannels):
    # Any that are 6 won't be in final output
    outchannels = [x for x in srcchannels if x != 6]
    # Any channel request that is higher than channels available, just use channel 0
    # (This also deals with channels we will later fill with zero or one)
    cpychannels = [x if x < tensor.shape[1] else 0 for x in outchannels]

    # Copy the desired source channel into place (or the first channel if we will replace in the next step)
    tensor = tensor[:, cpychannels]

    # Replace any channels with 0 or 1 if requested
    for i, c in enumerate(outchannels):
        if c == 4:
            tensor[:, i] = torch.zeros_like(tensor[0][i])
        elif c == 5:
            tensor[:, i] = torch.ones_like(tensor[0][i])

    return tensor


def gaussianblur(tensor, sigma):
    if np.isscalar(sigma):
        sigma = (sigma, sigma)
    kernel = [ceil(sigma[0] * 6), ceil(sigma[1] * 6)]
    kernel = [kernel[0] - kernel[0] % 2 + 1, kernel[1] - kernel[1] % 2 + 1]
    return torchvision.transforms.functional.gaussian_blur(tensor, kernel, sigma)


def directionalblur(tensor, sigma, direction: Literal["up", "down"], repeat_count=256):
    orig = tensor
    sigma /= sqrt(repeat_count)

    for _ in range(repeat_count):
        tensor = gaussianblur(tensor, sigma)
        if direction == "down":
            tensor = torch.minimum(tensor, orig)
        else:
            tensor = torch.maximum(tensor, orig)

    return tensor


def crop(tensor, top, left, height, width):
    return tensor[:, :, top : top + height, left : left + width]


def scale_shape(shape: tuple[int, ...], scale: float) -> tuple[int, ...]:
    return shape[:-2] + (round(shape[-2] * scale), round(shape[-1] * scale))


def get_dim(tensor) -> tuple[int, int]:
    return tensor.shape[-2:]


def resize(tensor, factors: float | tuple[float] | tuple[float, float], sharpness=1):
    if not isinstance(factors, typing.Iterable):
        factors = (factors, factors)
    elif len(factors) == 1:
        factors = (factors[0], factors[0])

    # Handle sharpness == 1 (lanczos3 for both up & downsampling, with antialiasing)
    # and sharpness == 2 (no antialiasing)

    if sharpness > 0:
        return resize_right(
            tensor,
            factors,
            interp_method=interp_methods.lanczos3,
            antialiasing=sharpness == 1,
            pad_mode="reflect",
        ).clamp(0, 1)

    # Handle sharpness = 0 (lanzcos3 for up, area for downscale)

    factors = tuple(factors)

    up_factors = tuple((1 if x < 1 else x for x in factors))
    down_factors = tuple((1 if x > 1 else x for x in factors))

    if up_factors != (1, 1):
        tensor = resize_right(
            tensor,
            up_factors,
            interp_method=interp_methods.lanczos3,
            antialiasing=False,
            pad_mode="reflect",
        ).clamp(0, 1)

    if down_factors != (1, 1):
        tensor = kornia.geometry.transform.rescale(
            tensor,
            down_factors,
            "area",
            antialias=False,
        ).clamp(0, 1)

    return tensor


def rescale(
    tensor,
    height,
    width=None,
    fit: Literal["strict", "cover", "contain"] = "cover",
    pad_mode="constant",
    sharpness=1,
):
    if width is None:
        width = height

    # Get the original height and width
    orig_h, orig_w = tensor.shape[-2], tensor.shape[-1]
    # Calculate the scaling factors to hit the target height
    scale_h, scale_w = height / orig_h, width / orig_w

    if fit == "cover":
        scale_h = scale_w = max(scale_h, scale_w)
    elif fit == "contain":
        scale_h = scale_w = min(scale_h, scale_w)

    # Do the resize
    tensor = resize(tensor, (scale_h, scale_w), sharpness=sharpness)

    # Get the result height and width
    res_h, res_w = tensor.shape[-2], tensor.shape[-1]

    # Calculate the "half error", negative is too much, positive is too little
    err_h, err_w = (height - res_h) // 2, (width - res_w) // 2

    # Crop any extra off
    tensor = crop(
        tensor, -err_h if err_h < 0 else 0, -err_w if err_w < 0 else 0, height, width
    )

    # Pad any missing
    pad = [err_w, width - res_w - err_w] if err_w > 0 else [0, 0]
    pad += [err_h, height - res_h - err_h] if err_h > 0 else [0, 0]

    return torch.nn.functional.pad(tensor, pad, pad_mode)


def canny_edge(tensor, low_threshold, high_threshold):
    cvi = toCV(tensor)
    res = []

    for i in range(cvi.shape[0]):
        cvii = cvi[i]
        edges = cv.Canny(cvii, low_threshold, high_threshold)
        res.append(fromCV(edges))

    return torch.cat(res, dim=0).to(tensor)


# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super().__init__()
        self.l2_term = torch.nn.MSELoss(reduction="mean")
        self.regularization_term = kornia.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(
            data=noisy_image.clone(), requires_grad=True
        )
        self.noisy_image = noisy_image

    def forward(self):
        res = self.l2_term(
            self.clean_image, self.noisy_image
        ) + 0.0001 * self.regularization_term(self.clean_image)

        return res

    def get_clean_image(self):
        return self.clean_image


def denoise(tensor, max_loss=None, iter_min=200, iter_max=5000):
    denoiser = TVDenoise(tensor)

    # define the optimizer to optimize the 1 parameter of tv_denoiser
    optimizer = torch.optim.SGD(denoiser.parameters(), lr=0.1, momentum=0.9)

    progress = tqdm(range(iter_max), desc="Denoising")
    for i in progress:
        optimizer.zero_grad()
        loss = denoiser().sum()
        if i % 50 == 0:
            progress.set_postfix_str(f"Loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()

        i = i + 1

        if i >= iter_max:
            break
        if i > iter_min and max_loss is not None and loss.item() < max_loss:
            break

    return denoiser.get_clean_image()


def normalmap_from_depthmap(
    depthmap: torch.Tensor,
    mask: torch.Tensor | None = None,
    background_threshold=0.1,
    a=np.pi * 2.0,
    preblur=None,
    postblur=None,
    smoothing=None,
    mode: Literal["alpha", "unit", "zero"] = "alpha",
):
    if preblur:
        depthmap_blr = kornia.filters.median_blur(depthmap, (preblur, preblur))
    else:
        depthmap_blr = depthmap

    edges = kornia.filters.spatial_gradient(depthmap_blr, normalized=False)

    # unpack the edges
    grad_x = edges[:, :, 0]
    grad_y = edges[:, :, 1]
    grad_z = torch.ones_like(grad_x) * a

    # If we have a background threshold, convert (or build from depth) a binary mask
    if background_threshold:
        if mask is None:
            mask = normalise_range(depthmap)

        ones = torch.ones_like(grad_x)
        zeros = torch.zeros_like(grad_x)

        mask = torch.where(mask < background_threshold, zeros, ones)

    # Now, if we have a mask, apply it
    if mask is not None:
        grad_x = grad_x * mask
        grad_y = grad_y * mask
        if mode == "zero":
            grad_z = grad_z * mask

    # Concat into single BCHW normal map
    normalmap = torch.cat((grad_x, grad_y, torch.ones_like(grad_x) * a), dim=1)

    # Normalise vector length
    veclen = (normalmap**2.0).sum(dim=1) ** 0.5
    normalmap = normalmap / veclen

    normalmap = (normalmap + 1) / 2

    if postblur:
        normalmap = kornia.filters.median_blur(normalmap, (postblur, postblur))

    if smoothing:
        # The worst contouring occurs on surfaces that are flat & parallel to the screen
        # Fortunately (because of vector normalisation) Z axis will be 1 when flat, and 0 when oblique
        # So we weight the denoising by the Z axis (blurred and normalised)
        weights = normalmap[:, [2]]
        weights = kornia.filters.box_blur(weights, (13, 13))
        weights = kornia.filters.median_blur(weights, (13, 13))
        weights = normalise_range(weights)

        if False:  # Debug weights
            return weights[:, [0, 0, 0]]

        denoised = denoise(normalmap)
        normalmap = normalmap + (denoised - normalmap) * weights * smoothing

    if mode == "alpha" and mask is not None:
        normalmap = torch.cat((normalmap, mask), dim=1)

    return normalmap


def blend_frequency_split(tensor_high, tensor_low, sigma, endsig, steps=None):
    if steps is None:
        steps = math.ceil((sigma - endsig) * 2)

    result = torch.zeros_like(tensor_high)
    high_prev = low_prev = None

    for i in np.linspace(0, 1, steps):
        stepsig = endsig + (sigma - endsig) * (1 - i)

        if stepsig == endsig:
            result += tensor_high - high_prev

        else:
            high_lp = gaussianblur(tensor_high, stepsig)
            low_lp = gaussianblur(tensor_low, stepsig)

            high_frag = (high_lp - high_prev) if high_prev is not None else high_lp
            low_frag = (low_lp - low_prev) if low_prev is not None else low_lp

            result = result + low_frag * (1 - i) + high_frag * i

            high_prev = high_lp
            low_prev = low_lp

    return result.clamp(0, 1)


def blend_frequency_split_1(tensor_high, tensor_low, sigma):
    high_hf = tensor_high - gaussianblur(tensor_high, sigma)
    low_lf = gaussianblur(tensor_low, sigma)

    return (low_lf + high_hf).clamp(0, 1)


def palletize(tensor, colours):
    down_factors = (colours / tensor.shape[-2], colours / tensor.shape[-1])

    downscaled = kornia.geometry.transform.rescale(
        tensor,
        down_factors,
        "area",
        antialias=False,
    ).clamp(0, 1)

    result = kornia.geometry.transform.resize(
        downscaled, tensor.shape[-2:], "nearest"
    ).clamp(0, 1)

    return result


def quantize(tensor, thresholds):
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), "Quantize only works on tensors in range 0..1"

    targets = np.linspace(0, 1, len(thresholds) + 1)[:-1]

    prev = 0
    for threshold, target in zip(thresholds, targets):
        tensor = torch.where((tensor > prev) & (tensor <= threshold), target, tensor)
        prev = target

    tensor = torch.where(tensor > prev, 1, tensor)

    return tensor


def shuffle(tensor):
    cv_images = toCV(tensor)
    results = []

    for image in cv_images:
        shuffler = ContentShuffleDetector()
        cv_result = shuffler(image)
        results.append(fromCV(cv_result))

    return torch.cat(results, dim=0).to(tensor)


def information_in_alpha(tensor):
    if tensor.shape[1] < 4:
        return False

    if len(unique := tensor.unique_consecutive()) == 1:
        return unique == 1.0

    # TODO: Which is faster?
    # if unique.mean() == 1.0 and unique.std() == 0.0:
    #     return False

    return True


def infill(tensor, mask, size, step=2):
    """mask should be 0 keep, 1 replace"""

    # Pad tensor and mask, to avoid rotation on shift
    tensor = torch.nn.functional.pad(tensor, (size, size, size, size), mode="replicate")
    mask = torch.nn.functional.pad(mask, (size, size, size, size), mode="replicate")

    # Slightly expand, then harden, then invert mask
    mask = gaussianblur(mask, 1)
    mask = quantize(mask, [0.01])
    mask = 1 - mask

    # Result and mask accumulator
    result = tensor * mask
    mask_accum = mask

    for s in range(step, size, step):
        for rx in (-s, 0, s):
            for ry in (-s, 0, s):
                shifted = tensor.roll(shifts=(ry, rx), dims=(-2, -1))
                shifted_mask = mask.roll(shifts=(ry, rx), dims=(-2, -1))

                result = result + shifted * shifted_mask * (1 - mask_accum)
                mask_accum = (mask_accum + shifted_mask).clamp(0, 1)

    s = result.shape
    return result[:, :, size : s[-2] - size, size : s[-1] - size]


def match_histograms(image, reference):
    image_cv = toCV(image)
    reference_cv = toCV(reference)

    result = np_match_histograms(image_cv, reference_cv, channel_axis=3)
    return fromCV(result).to(image)
