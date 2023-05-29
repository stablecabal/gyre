import gc
import inspect
import logging
from typing import Iterable, Optional, Union

import generation_pb2
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import deprecate
from tqdm.auto import tqdm

from gyre.pipeline import pipeline_meta
from gyre.pipeline.model_utils import GPUExclusionSet, clone_model
from gyre.pipeline.prompt_types import HintImage, ImageLike, PromptBatchLike
from gyre.pipeline.samplers import build_sampler_set
from gyre.pipeline.unified_pipeline import SCHEDULER_NOISE_TYPE
from gyre.pipeline.xformers_utils import xformers_mea_available

logger = logging.getLogger(__name__)


class ProgressBarAbort(BaseException):
    pass


class ProgressBarWrapper(object):
    class InternalTqdm(tqdm):
        def __init__(self, pipeline, progress_callback, stop_event, *args, **kwargs):
            self._pipeline = pipeline
            self._progress_callback = progress_callback
            self._stop_event = stop_event
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            displayed = super().update(n)

            # If we've been aborted, throw the ProgressBarAbort exception
            # This should be caught by either the __iter__ or __exit__ overrides below
            if self._stop_event and self._stop_event.is_set():
                self.set_description("ABORTED")
                raise ProgressBarAbort()

            # If there's a callback to update, call it
            if displayed and self._progress_callback:
                self._progress_callback(**self.format_dict)

            return displayed

        @property
        def format_dict(self):
            return super().format_dict | {
                "pipeline_id": self._pipeline.id,
                "pipeline_unique_id": id(self._pipeline),
            }

    def __init__(
        self,
        pipeline,
        progress_callback,
        stop_event,
        suppress_output=False,
        **extra_args,
    ):
        self._pipeline = pipeline
        self._progress_callback = progress_callback
        self._stop_event = stop_event
        self._suppress_output = suppress_output
        self._extra_args = extra_args

    def __call__(self, iterable=None, total=None):
        args = [self._pipeline, self._progress_callback, self._stop_event]
        kwargs = {"disable": self._suppress_output, **self._extra_args}

        if iterable is not None:
            args.append(iterable)
        elif total is not None:
            kwargs["total"] = total

        return ProgressBarWrapper.InternalTqdm(*args, **kwargs)


class PipelineWrapper:
    def __init__(self, id, mode, pipeline):
        self._id = id
        self._mode = mode

        self._pipeline = pipeline
        self._previous = None

    @property
    def id(self):
        return self._id

    @property
    def mode(self):
        return self._mode

    def pipeline_modules(self):
        pipeline_module_helper = getattr(self._pipeline, "pipeline_modules", None)

        if pipeline_module_helper:
            for name, module in pipeline_module_helper():
                yield self._pipeline, name, module

        else:
            module_names, *_ = self._pipeline.extract_init_dict(
                dict(self._pipeline.config)
            )
            for name in module_names.keys():
                module = getattr(self._pipeline, name)
                if isinstance(module, torch.nn.Module):
                    yield self._pipeline, name, module

    def activate(self, device, exclusion_set=None):
        if self._previous is not None:
            raise Exception("Activate called without previous deactivate")

        self._previous = []

        for source, name, module in self.pipeline_modules():
            self._previous.append((source, name, module))

            # Clone from CPU to either CUDA or Meta with a hook to move to CUDA
            cloned = clone_model(
                module,
                device,
                exclusion_set=exclusion_set,
            )

            # And set it on the pipeline
            setattr(source, name, cloned)

        hintset_manager = getattr(self._pipeline, "hintset_manager", None)
        self._previous_hintset_manager = hintset_manager
        if hintset_manager:
            self._pipeline.hintset_manager = hintset_manager.with_aligner(
                lambda hint: clone_model(
                    hint,
                    device,
                    exclusion_set=exclusion_set,
                )
            )

    def deactivate(self):
        if self._previous is None:
            raise Exception("Deactivate called without previous activate")

        for source, name, module in self._previous:
            setattr(source, name, module)

        self._pipeline.hintset_manager = self._previous_hintset_manager

        self._previous = None
        self._previous_hintset_manager = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        return self._pipeline(*args, **kwargs)


class DiffusionPipelineWrapper(PipelineWrapper):
    def __init__(self, id, mode, pipeline):
        super().__init__(id, mode, pipeline)

        # Get any pipeline-specific tweaks
        self._meta = pipeline_meta.get_meta(pipeline)

        # Enable attention slicing based on mode
        if self.mode.attention_slice:
            self._pipeline.enable_attention_slicing("auto")
            if callable(getattr(self._pipeline, "enable_vae_slicing", None)):
                self._pipeline.enable_vae_slicing()
        else:
            self._pipeline.disable_attention_slicing()
            if callable(getattr(self._pipeline, "disable_vae_slicing", None)):
                self._pipeline.disable_vae_slicing()

        # Enable VAE tiling based on mode
        if (vae := getattr(self._pipeline, "vae", None)) is not None:
            if self.mode.tile_vae:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

        # Enable xformers based on mode
        if self._meta.get("xformers") == "manual":
            if xformers_mea_available():
                self._pipeline.enable_xformers_memory_efficient_attention()

        # If the pipeline has a scheduler, get some details
        self._prediction_type = None
        self._samplers = None

        if (scheduler := getattr(self._pipeline, "scheduler", None)) is not None:
            self._prediction_type = scheduler.config.get("prediction_type", "epsilon")

            self._samplers = build_sampler_set(
                scheduler.config,
                include_diffusers=self._meta.get("diffusers_capable", True),
                include_kdiffusion=self._meta.get("kdiffusion_capable", False),
            )

    def _prepScheduler(self, scheduler):
        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        return scheduler

    def activate(self, device):
        exclusion_set = None

        if self.mode.gpu_offload and self._meta.get("offload_capable", True):
            exclusion_set = GPUExclusionSet(
                name=str(device).upper(),
                max_activated=self.mode.model_max_limit,
                mem_limit=self.mode.model_vram_limit,
            )

        super().activate(device, exclusion_set)

    def get_samplers(self):
        return self._samplers

    def _build_generator(self, seed: int | Iterable[int] | None):
        generator = None

        generator_device = "cpu" if self.mode.device == "mps" else self.mode.device

        if isinstance(seed, Iterable):
            generator = [torch.Generator(generator_device).manual_seed(s) for s in seed]
        elif seed is not None and seed > 0:
            generator = torch.Generator(generator_device).manual_seed(seed)

        return generator

    def _inject_scheduler(self, sampler, scheduler):
        # Inject custom scheduler
        if (samplers := self.get_samplers()) is not None:
            if scheduler is None:
                if sampler is None:
                    scheduler = list(samplers.values())[0]
                else:
                    scheduler = samplers.get(sampler, None)

            if not scheduler:
                raise NotImplementedError("Scheduler not implemented")

            self._pipeline.scheduler = scheduler

    def _filter_args(self, **pipeline_args):
        # Allow meta-info to adjust fields for the specific pipeline
        pipeline_args = pipeline_meta.apply_fields(self._meta, pipeline_args)

        # Introspect some details about the pipeline call metiod
        pipeline_keys = inspect.signature(self._pipeline).parameters.keys()
        self_params = inspect.signature(self.__call__).parameters

        # Filter args to only those the pipeline supports
        for k, v in list(pipeline_args.items()):
            if k not in pipeline_keys:
                if not (k in self_params and v == self_params[k].default):
                    print(
                        f"Warning: Pipeline doesn't understand argument {k} (set to {v}) - ignoring"
                    )
                del pipeline_args[k]

        return pipeline_args

    def __call__(
        self,
        # The prompt, negative_prompt, and number of images per prompt
        prompt: PromptBatchLike = "",
        negative_prompt: PromptBatchLike | None = None,
        num_images_per_prompt: Optional[int] = 1,
        # The seeds - len must match len(prompt) * num_images_per_prompt if provided
        seed: Optional[Union[int, Iterable[int]]] = None,
        # The size - ignored if an image is passed
        height: int = 512,
        width: int = 512,
        # Guidance control
        guidance_scale: float = 7.5,
        clip_guidance_scale: Optional[float] = None,
        clip_guidance_base: Optional[str] = None,
        # Sampler control
        sampler: generation_pb2.DiffusionSampler = None,
        scheduler=None,
        eta: Optional[float] = None,
        churn: Optional[float] = None,
        churn_tmin: Optional[float] = None,
        churn_tmax: Optional[float] = None,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        karras_rho: Optional[float] = None,
        scheduler_noise_type: Optional[SCHEDULER_NOISE_TYPE] = "normal",
        num_inference_steps: int = 50,
        # Providing these changes from txt2img into either img2img (no mask) or inpaint (mask) mode
        image: ImageLike | None = None,
        mask_image: ImageLike | None = None,
        outmask_image: ImageLike | None = None,
        depth_map: ImageLike | None = None,
        hint_images: list[HintImage] | None = None,
        # The strength of the img2img or inpaint process, if image is provided
        strength: float = None,
        # Lora
        lora=None,
        # Token Embeddings
        token_embeddings=None,
        # Hires control
        hires_fix=None,
        hires_oos_fraction=None,
        # Tiling control
        tiling=False,
        # Debug control
        debug_latent_tags=None,
        debug_latent_prefix="",
        # Process control
        progress_callback=None,
        stop_event=None,
        suppress_output=False,
    ):
        # Convert seeds to generators
        generator = self._build_generator(seed)

        self._inject_scheduler(sampler, scheduler)

        # Inject progress bar to enable cancellation support
        self._pipeline.progress_bar = ProgressBarWrapper(
            self, progress_callback, stop_event, suppress_output
        )

        pipeline_args = dict(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            cfg_execution=self.mode.cfg_execution,
            clip_guidance_scale=clip_guidance_scale,
            clip_guidance_base=clip_guidance_base,
            prediction_type=self._prediction_type,
            eta=eta,
            churn=churn,
            churn_tmin=churn_tmin,
            churn_tmax=churn_tmax,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            karras_rho=karras_rho,
            scheduler_noise_type=scheduler_noise_type,
            num_inference_steps=num_inference_steps,
            image=image,
            mask_image=mask_image,
            outmask_image=outmask_image,
            depth_map=depth_map,
            hint_images=hint_images,
            strength=strength,
            lora=lora,
            token_embeddings=token_embeddings,
            hires_fix=hires_fix,
            hires_oos_fraction=hires_oos_fraction,
            tiling=tiling,
            debug_latent_tags=debug_latent_tags,
            debug_latent_prefix=debug_latent_prefix,
            output_type="tensor",
            return_dict=False,
        )

        pipeline_args = self._filter_args(**pipeline_args)

        try:
            images = self._pipeline(**pipeline_args)
        except ProgressBarAbort:
            images = None

        gc.collect()

        return images

    def generate(self, *args, **kwargs):
        logger.warn("UnifiedPipeline#generate is deprecated, use __call__")
        return self(*args, **kwargs)
