import gc
import logging
from typing import Iterable, Optional, Union

import torch
from diffusers.models import modeling_utils
from PIL.Image import Image as PILImage
from tqdm.auto import tqdm

from gyre import images
from gyre.pipeline import pipeline_meta
from gyre.pipeline.pipeline_wrapper import (
    DiffusionPipelineWrapper,
    ProgressBarAbort,
    ProgressBarWrapper,
)
from gyre.pipeline.prompt_types import ImageLike, PromptBatchLike
from gyre.pipeline.upscalers import utils as upscaler_utils

logger = logging.getLogger(__name__)


class DiffusionUpscalerPipelineWrapper(DiffusionPipelineWrapper):
    def __init__(self, id, mode, pipeline):
        super().__init__(id, mode, pipeline)
        pipeline.decode_latents = self.decode_latents

        # Upscaler-4x always enables tiling, even on 24GB cards, it's brutal on memory without it
        pipeline.vae.enable_tiling()

    # A version of decode_latents that just returns the tensor directly
    def decode_latents(self, latents):
        latents = 1 / self._pipeline.vae.config.scaling_factor * latents
        image = self._pipeline.vae.decode(latents).sample
        image = image / 2 + 0.5
        # Better results if we only clamp later after frequency merge
        # image = image.clamp(0, 1)
        return image

    def _simple_upscale(self, image, scale=4, tile_size=512):
        simple_input = images.normalise_tensor(image)

        return upscaler_utils.tile(
            simple_input,
            lambda tile: images.resize(tile, scale),
            scale,
            tile_size,
            # Progress bar kwargs
            disable=True,
        )

    def _pipeline_upscale(
        self,
        image,
        num_images_per_prompt,
        generator,
        scale=4,
        tile_size=512,
        **kwargs,
    ):
        # The pipeline args that are consistent for every call
        pipeline_args = dict(
            **kwargs,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="numpy",  # Actually "tensor" due to decode_latents patch above
            return_dict=False,
        )

        # Generate the image noise once, so tile overlaps are consistent
        image_noise = self._pipeline.prepare_latents(
            *image.shape, image.dtype, image.device, generator
        )

        # And same deal with the latents
        batch_total = image.shape[0] * num_images_per_prompt
        latents = self._pipeline.prepare_latents(
            batch_total, 4, *image.shape[-2:], image.dtype, image.device, generator
        )

        # Cat them all together for passing into the tiler
        noise_offset = image.shape[0]
        latents_offset = noise_offset + image_noise.shape[0]
        tiler_input = torch.cat([image, image_noise, latents], dim=0)

        # Get the current add_noise method. We need to patch it to return our custom noise
        orig_noise = self._pipeline.low_res_scheduler.add_noise

        # Build the callback per tile
        def tiled_callback(tile):
            # Split the tiled data back into thte image, image noise and latent noise tensors
            image = images.normalise_tensor(tile[:noise_offset], 3)
            noise = images.normalise_tensor(tile[noise_offset:latents_offset], 3)
            latents = tile[latents_offset:]

            try:
                # Patch the add_noise method to use our custom noise
                self._pipeline.low_res_scheduler.add_noise = (
                    lambda image, _, noise_level: orig_noise(image, noise, noise_level)
                )

                # Call the pipeline and return the image result
                return self._pipeline(
                    image=pipeline_meta.image_to_centered(image) * 0.9,
                    latents=latents,
                    **pipeline_args,
                )[0]

            finally:
                # Remove the patch - this restores the class-based method
                del self._pipeline.low_res_scheduler.add_noise

        try:
            return upscaler_utils.tile(
                tiler_input, tiled_callback, scale, tile_size, desc="Tiling upscaler"
            )
        except ProgressBarAbort:
            return None

    def __call__(
        self,
        # The image
        image: ImageLike,
        # The prompt, negative_prompt, and number of images per prompt
        prompt: PromptBatchLike = "",
        negative_prompt: PromptBatchLike | None = None,
        num_images_per_prompt: int = 1,
        # The seeds - len must match len(prompt) * num_images_per_prompt if provided
        seed: Optional[Union[int, Iterable[int]]] = None,
        # The size - ignored if an image is passed
        height: int | None = 512,
        width: int | None = 512,
        # Sampler control
        num_inference_steps: int = 50,
        # Process control
        progress_callback=None,
        stop_event=None,
        suppress_output=False,
        **_,
    ):
        # Convert seeds to generators
        generator = self._build_generator(seed)

        # Inject progress bar to enable cancellation support
        self._pipeline.progress_bar = ProgressBarWrapper(
            self,
            progress_callback,
            stop_event,
            suppress_output,
            desc="    Tile: ",
            leave=False,
        )

        # Prepare the image
        device = self._pipeline._execution_device
        dtype = modeling_utils.get_parameter_dtype(self._pipeline.unet)

        image = images.fromPIL(image) if isinstance(image, PILImage) else image
        image = images.normalise_tensor(image, 4).to(device, dtype)

        # Check height & width
        if height == image.shape[-2] and width == image.shape[-1]:
            logger.debug("Ignoring passed size, since it matches input image")
            height = width = None

        # --- SIMPLE UPSCALE

        simple_upscale = self._simple_upscale(image, 4, 512)

        gc.collect()

        # --- DIFFUSION UPSCALE

        # Prepare the pipline_upscale args
        kwargs = self._filter_args(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
        )

        # And get the upscale
        diffusion_upscale = self._pipeline_upscale(image, **kwargs, generator=generator)

        gc.collect()

        # If we aborted, just return None
        if diffusion_upscale is None:
            return None

        # --- FREQUENCY MERGE

        # When tiling, the result is inconsistent lightness between tiles, presumably
        # because the VAE process has some level of normalisation. So we only use the
        # _high frequency_ component of the upscaler-4x result, and for the low frequency
        # we just use a regular lanczos3 upscale of the whole image

        SIGMA_START, SIGMA_END = 3, 1  # Magic numbers, determined experimentally

        result = images.blend_frequency_split(
            diffusion_upscale, simple_upscale, SIGMA_START, SIGMA_END
        )

        gc.collect()

        # --- DOWNSCALE

        if width is not None and height is not None:
            if width > result.shape[-1] or height > result.shape[-2]:
                logger.warn(
                    f"Requested size ({width}x{height}) is greated than result after upscale ({result.shape[-2]}x{result.shape[-1]}). "
                    "Further lower-quality upscaling will occur."
                )

            result = images.rescale(result, height, width, "cover")

        return result
