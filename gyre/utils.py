import io

import cv2 as cv
import generation_pb2
import numpy as np
import PIL
import torch

from gyre import images
from gyre.pipeline.vae_approximator import VaeApproximator


def artifact_to_image(artifact):
    if (
        artifact.type == generation_pb2.ARTIFACT_IMAGE
        or artifact.type == generation_pb2.ARTIFACT_MASK
    ):
        img = PIL.Image.open(io.BytesIO(artifact.binary))
        return img
    else:
        raise NotImplementedError("Can't convert that artifact to an image")


def image_to_artifact(im, artifact_type=generation_pb2.ARTIFACT_IMAGE, meta=None):
    binary = None

    # Handle tensor
    if isinstance(im, torch.Tensor):
        if im.ndim == 4 and im.shape[0] > 1:
            raise ValueError("Can't convert batches of >1 image")
        binary = images.toPngBytes(im)[0]

    # Handle PIL image
    elif isinstance(im, PIL.Image.Image):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        binary = buf.getvalue()

    # Handle nparray in various forms (0..1, 0..255, BCHW / BHWC / CHW / HWC format)
    elif isinstance(im, np.ndarray):
        if im.ndim == 4 and im.shape[0] > 1:
            raise ValueError("Can't convert batches of >1 image")

        if im.ndim == 3:
            im = im[None, ...]

        if im.shape[1] < 16:
            im = np.moveaxis(im, 1, 3)

        if im.dtype != np.uint8:
            im = (im * 255).round().astype("uint8")

        # Don't forget to convert to bgr
        binary = cv.imencode(".png", im[0][:, :, [2, 1, 0]])[1].tobytes()

    # What did you give me?
    else:
        raise ValueError(
            f"Don't know how to convert image of type {type(im)} to a png binary"
        )

    if meta:
        for k, v in meta.items():
            binary = images.addTextChunkToPngBytes(binary, k, v)

    return generation_pb2.Artifact(type=artifact_type, binary=binary, mime="image/png")


class CallbackImageWrapper:
    def __init__(self, callback, device, dtype):
        self.callback = callback
        self.vae_approximator = VaeApproximator()

    def __call__(self, i, t, latents):
        pixels = self.vae_approximator(latents)
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        self.callback(i, t, pixels)
