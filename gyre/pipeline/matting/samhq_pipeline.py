import logging
from typing import Literal

import torch
from diffusers.models import modeling_utils
from diffusers.pipeline_utils import DiffusionPipeline
from PIL.Image import Image as PILImage
import numpy as np
import cv2

from gyre.logging import VisualRecord as vr

from gyre import images
from gyre.pipeline.prompt_types import ImageLike, LocationsOfInterest

from segment_anything import SamPredictor

logger = logging.getLogger(__name__)


class SamHQPipeline:

    samhq: torch.nn.Module
    vitmatte: torch.nn.Module

    def __init__(self, samhq, vitmatte):
        super().__init__()
        self.samhq = samhq
        self.vitmatte = vitmatte

    def to(self, device):
        self.samhq.to(device)
        self.vitmatte.to(device)

    def pipeline_modules(self):
        return [("samhq", self.samhq), ("vitmatte", self.vitmatte)]

    def erode(self, mask, erode_size):
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size)
        )
        return cv2.erode(mask, erode_kernel, iterations=5)

    def dilate(self, mask, dilate_size):
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
        )
        return cv2.dilate(mask, dilate_kernel, iterations=5)

    def generate_trimap(self, mask, erode_kernel_size=10, dilate_kernel_size=10):
        trimap = np.zeros_like(mask, dtype=np.float32)

        if dilate_kernel_size:
            dilated = self.dilate(mask, dilate_kernel_size)
            trimap[dilated == 255] = 0.5

        if erode_kernel_size:
            eroded = self.erode(mask, erode_kernel_size)
            trimap[eroded == 255] = 1.0
        else:
            trimap[mask == 255] = 1.0

        return trimap

    @torch.no_grad()
    def __call__(
        self,
        image: ImageLike,
        output_type: Literal["mask", "overmask", "undermask", "trimap", "matte"],
        lois: LocationsOfInterest,
        erode: int = 10,
        dilate: int = 10,
        **kwargs,
    ):
        predictor = SamPredictor(self.samhq)

        if isinstance(image, PILImage):
            image = images.fromPIL(image)

        image = images.normalise_tensor(image, 3)

        # -- Use samhq to product a binary mask from locations of interest

        sdevice = modeling_utils.get_parameter_device(self.samhq)
        sdtype = modeling_utils.get_parameter_dtype(self.samhq)

        sample = predictor.transform.apply_image_torch(image.to(sdevice, sdtype) * 255)
        predictor.set_torch_image(sample, (image.shape[2], image.shape[3]))

        points = []
        labels = []
        rect = None

        if lois.rectangles:
            if len(lois.rectangles) > 1:
                raise ValueError("Can only accept at most one rectangle LoI")

            loi = lois.rectangles[0]
            rect = [loi.left, loi.top, loi.right, loi.bottom]

        for loi in lois.points:
            points.append([loi.x, loi.y])
            labels.append(loi.label)

        masks, _, _ = predictor.predict(
            point_coords=np.array(points) if points else None,
            point_labels=np.array(labels) if labels else None,
            box=np.array(rect) if rect else None,
            # return_logits=True
        )

        mask = masks[0].astype(np.float32)

        if output_type == "overmask":
            mask = self.dilate(mask, dilate)
        elif output_type == "undermask":
            mask = self.erode(mask, erode)

        mask_tensor = torch.from_numpy(mask).unsqueeze(dim=0).unsqueeze(dim=0)

        if output_type in {"mask", "overmask", "undermask"}:
            return mask_tensor.to(image)
        else:
            logger.debug(vr("Binary mask {mask}", mask=mask_tensor))

        # -- From the mask, produce a trimap

        mask = mask.astype(np.uint8) * 255

        trimap = self.generate_trimap(mask)
        trimap_tensor = torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0)

        if output_type == "trimap":
            return trimap_tensor.to(image)
        else:
            logger.debug(vr("Trimap {trimap}", trimap=trimap_tensor))

        # -- Run the trimap through vitmatte to product a matte

        vdevice = modeling_utils.get_parameter_device(self.vitmatte)
        vdtype = modeling_utils.get_parameter_dtype(self.vitmatte)

        input = {
            "image": image.to(vdevice, vdtype)[:, [2, 1, 0]],
            "trimap": trimap_tensor.to(vdevice, vdtype),
        }

        alpha = self.vitmatte(input)["phas"]

        # Slightly shrink mask
        alpha = ((alpha - 0.2) / 0.8).clamp(0, 1)

        # And done
        return alpha.to(image)
