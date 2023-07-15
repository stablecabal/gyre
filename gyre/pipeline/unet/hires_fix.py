from typing import cast

import torch
import torchvision.transforms as T
import logging

from gyre import resize_right, images
from gyre.logging import VisualRecord as vr
from gyre.pipeline.easing import Easing
from gyre.pipeline.randtools import batched_rand
from gyre.pipeline.unet.types import (
    DiffusersSchedulerUNet,
    GenericSchedulerUNet,
    KDiffusionSchedulerUNet,
    PX0Tensor,
    XtTensor,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Indexes into a shape for the height and width dimensions
# Negative indexed to work for any number of dimensions
Hi, Wi = -2, -1


def pad_like(latents, like, mode="replicate"):
    wd = like.shape[Wi] - latents.shape[Wi]
    hd = like.shape[Hi] - latents.shape[Hi]
    l = wd // 2
    r = wd - l
    t = hd // 2
    b = hd - t

    pad = torch.nn.functional.pad

    if isinstance(mode, int | float):
        return pad(latents, pad=(l, r, t, b), mode="constant", value=mode)
    else:
        return pad(latents, pad=(l, r, t, b), mode=mode)


def resize_nearest(latents, scale_factor=1):
    hs = int(latents.shape[Hi] * scale_factor)
    ws = int(latents.shape[Wi] * scale_factor)

    return T.functional.resize(latents, [hs, ws], T.InterpolationMode.NEAREST)


def scale_into(latents, scale, target=None, target_shape=None, mode="lanczos"):
    if mode == "nearest":
        latents = resize_nearest(latents, scale)
    else:
        latents = images.resize(latents, scale, sharpness=-1, clamp=False)

    if target is not None and target_shape is not None:
        raise ValueError("Only provide one of target or target_shape")

    if target is None and target_shape is None:
        raise ValueError("Only exactly one of target or target_shape")

    if target_shape is None:
        target_shape = target.shape

    # Now crop off anything that's outside target shape, and offset if it's inside target shape

    # Positive is offset into the shape, negative is crop amount
    offh = (target_shape[Hi] - latents.shape[Hi]) // 2
    offw = (target_shape[Wi] - latents.shape[Wi]) // 2

    if offh < 0:
        latents = latents[:, :, -offh : -offh + target_shape[Hi], :]
        offh = 0

    if offw < 0:
        latents = latents[:, :, :, -offw : -offw + target_shape[Wi]]
        offw = 0

    if target is not None:
        target[
            :, :, offh : offh + latents.shape[Hi], offw : offw + latents.shape[Wi]
        ] = latents
    else:
        pad_w = [offw, (target_shape[Wi] - latents.shape[Wi]) - offw]
        pad_h = [offh, (target_shape[Hi] - latents.shape[Hi]) - offh]

        target = torch.nn.functional.pad(latents, pad_w + pad_h, mode="replicate")

    return target


def down_scale_factor(latents_shape, target_shape, oos_fraction):
    scales = target_shape[Hi] / latents_shape[Hi], target_shape[Wi] / latents_shape[Wi]
    scale_min, scale_max = min(*scales), max(*scales)

    return scale_min * oos_fraction + scale_max * (1 - oos_fraction)


def up_scale_factor(latents_shape, target_shape, oos_fraction):
    return 1 / down_scale_factor(target_shape, latents_shape, oos_fraction)


DOWN_STRATEGY = "pad"
UP_STRATEGY = "clone"


def scale_strategy(src, like, oos_fraction, strategy, factor_calc):

    scale_factor = factor_calc(src.shape, like.shape, oos_fraction)

    if strategy == "pad":
        return scale_into(src, scale_factor, target_shape=like.shape)
    else:
        if strategy == "clone":
            scaled = like.clone()
        else:
            scaled = torch.zeros_like(like)

        return scale_into(src, scale_factor, target=scaled)


class HiresUnetWrapper(GenericSchedulerUNet):
    def __init__(
        self,
        unet_natural: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        unet_hires: DiffusersSchedulerUNet | KDiffusionSchedulerUNet,
        generators: list[torch.Generator],
        natural_size: torch.Size,
        oos_fraction: float,
        latent_debugger,
    ):
        self.unet_natural = unet_natural
        self.unet_hires = unet_hires
        self.generators = generators
        self.natural_size = natural_size
        self.oos_fraction = oos_fraction

        self.easing = Easing(floor=0, start=0, end=0.667, easing="cubic")
        self.latent_debugger = latent_debugger

    def __call__(self, latents: XtTensor, __step, u: float) -> PX0Tensor | XtTensor:
        # Linear blend between base and graft
        p = self.easing.interp(u)

        lo_in, hi_in = latents.chunk(2)

        if isinstance(__step, torch.Tensor) and __step.shape:
            lo_t, hi_t = __step.chunk(2)
        else:
            lo_t = hi_t = __step

        self.latent_debugger.log(
            "hires_in", int(u * 1000), torch.cat([lo_in, hi_in], dim=3)[0:1]
        )

        hi = self.unet_hires(hi_in, hi_t, u=u)

        # Early out if we're passed the graft stage
        if p >= 0.999:
            return cast(type(hi), torch.concat([lo_in, hi]))

        *_, h, w = latents.shape
        th, tw = self.natural_size

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        lo_in = lo_in[:, :, offseth : offseth + th, offsetw : offsetw + tw]
        lo = self.unet_natural(lo_in, lo_t, u=u)

        # Downscale hi and merge into lo
        hi_downscaled = scale_strategy(
            hi, lo, self.oos_fraction, DOWN_STRATEGY, down_scale_factor
        )

        self.latent_debugger.log(
            "hires_lopre", int(u * 1000), torch.cat([lo, hi_downscaled], dim=3)[0:1]
        )

        randmap = batched_rand(lo.shape, self.generators, lo.device, lo.dtype)
        lo_merged = torch.where(randmap >= p, lo, hi_downscaled)

        # Upscale lo and merge it back into hi
        lo_upscaled = scale_strategy(
            lo, hi, self.oos_fraction, UP_STRATEGY, up_scale_factor
        )

        self.latent_debugger.log(
            "hires_hipre", int(u * 1000), torch.cat([lo_upscaled, hi], dim=3)[0:1]
        )

        randmap = batched_rand(hi.shape, self.generators, hi.device, hi.dtype)
        hi_merged = torch.where(randmap >= p, lo_upscaled, hi)

        # Expand lo back to full tensor size by wrapping with 0
        lo_expanded = torch.ones_like(hi_merged) * lo_merged.mean(
            dim=[2, 3], keepdim=True
        )
        lo_expanded[:, :, offseth : offseth + th, offsetw : offsetw + tw] = lo_merged

        self.latent_debugger.log(
            "hires", int(u * 1000), torch.cat([lo_expanded, hi_merged], dim=3)[0:1]
        )

        if False:
            logger.debug(
                vr(
                    "{hi} {lo} {hi_merged} {lo_expanded}",
                    hi=hi,
                    lo=lo,
                    lo_expanded=lo_expanded,
                    hi_merged=hi_merged,
                )
            )

        res = torch.concat([lo_expanded, hi_merged])
        return cast(type(hi), res)

    @classmethod
    def image_to_natural(
        cls,
        natural_size: int,
        image: torch.Tensor,
        oos_fraction: float,
        fill=None,
        fill_blend=None,
    ):
        target = None
        target_shape = [natural_size, natural_size]
        scale_factor = down_scale_factor(image.shape, target_shape, oos_fraction)

        fill_base = None
        if fill is not None:
            fill_base = fill([*image.shape[:-2], natural_size, natural_size])

        if fill is not None and fill_blend is None:
            target, target_shape = fill_base, None

        result = scale_into(
            image, scale_factor, target=target, target_shape=target_shape
        ).clamp(0, 1)

        if fill_blend is not None:
            assert fill_base is not None

            print(image.shape, scale_factor, natural_size)

            blend_map = torch.zeros([*image.shape[:-2], natural_size, natural_size])
            scale_into(torch.ones_like(image), scale_factor, target=blend_map)
            blend_map = images.directionalblur(blend_map, fill_blend, "up")

            blend_map = blend_map.to(result)
            fill_base = fill_base.to(result)

            orig = result
            result = result * blend_map + fill_base * (1 - blend_map)

        return result

    @classmethod
    def merge_initial_latents(cls, left, right):
        left_resized = torch.zeros_like(right)

        *_, th, tw = left.shape
        *_, h, w = right.shape

        offseth = (h - th) // 2
        offsetw = (w - tw) // 2

        left_resized[:, :, offseth : offseth + th, offsetw : offsetw + tw] = left
        # right[:, :, offseth : offseth + th, offsetw : offsetw + tw] = left
        return torch.concat([left_resized, right])

    @classmethod
    def split_result(cls, left, right):
        return right.chunk(2)[1]
