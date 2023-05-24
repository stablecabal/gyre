import dataclasses
import json
import logging
import random
import threading
import time
import traceback
import uuid
from functools import cached_property
from math import sqrt
from queue import Empty, Queue
from types import SimpleNamespace as SN
from typing import Any, Callable, Iterable, Literal

import generation_pb2
import generation_pb2_grpc
import grpc
import torch
from accept_types import get_best_match
from google.protobuf import json_format as pb_json_format

from gyre import constants, images
from gyre.cache import CacheLookupError
from gyre.debug_recorder import DebugNullRecorder
from gyre.logging import VisualRecord as vr
from gyre.manager import EngineNotFoundError
from gyre.pipeline.prompt_types import HintImage, HintPriority, Prompt, PromptFragment
from gyre.protobuf_safetensors import UserSafetensors, deserialize_safetensors
from gyre.protobuf_tensors import deserialize_tensor
from gyre.services.exception_to_grpc import exception_to_grpc

logger = logging.getLogger(__name__)


def buildDefaultMaskPostAdjustments():
    hardenMask = generation_pb2.ImageAdjustment()
    hardenMask.levels.input_low = 0
    hardenMask.levels.input_high = 0.05
    hardenMask.levels.output_low = 0
    hardenMask.levels.output_high = 1

    blur = generation_pb2.ImageAdjustment()
    blur.blur.sigma = 32
    blur.blur.direction = generation_pb2.DIRECTION_UP

    return [hardenMask, blur]


DEFAULT_POST_ADJUSTMENTS = buildDefaultMaskPostAdjustments()


def image_to_artifact(
    image,
    artifact_type=generation_pb2.ARTIFACT_IMAGE,
    accept=None,
    encoded_parameters=None,
):
    tensor = images.fromAuto(image)

    if tensor.shape[0] > 1:
        raise RuntimeError("Can't encode tensors with batch > 1")

    # See what types are available
    available_types = ["image/png"]
    if images.SUPPORTS_WEBP:
        available_types.append("image/webp")

    # Compare with what types are acceptable. image/png is always acceptable
    mimetype = get_best_match(accept, available_types)
    if mimetype is None:
        mimetype = "image/png"

    if mimetype == "image/webp":
        binary = images.toWebpBytes(tensor)[0]
        if encoded_parameters:
            binary = images.addTextChunkToWebpBytes(binary, b"ICMT", encoded_parameters)

    else:
        binary = images.toPngBytes(tensor)[0]
        if encoded_parameters:
            binary = images.addTextChunkToPngBytes(
                binary, "generation_parameters", encoded_parameters
            )

    return generation_pb2.Artifact(type=artifact_type, binary=binary, mime=mimetype)


class AsyncContext:
    def __init__(self, deadline=None):
        self.queue = Queue()
        self.code = grpc.StatusCode.OK
        self.message = ""
        self.cancel_callback: Callable | None = None
        self.thread: threading.Thread | None = None
        self.deadline: float | None = None

        if deadline:
            self.deadline = time.monotonic() + deadline

    # These are methods for the async handlers

    def cancel(self):
        if self.cancel_callback:
            self.cancel_callback()

        self.code = grpc.StatusCode.CANCELLED
        self.message = "Cancelled"

    def set_deadline(self, deadline):
        new_deadline = time.monotonic() + deadline

        if self.deadline:
            self.deadline = min(new_deadline, self.deadline)
        else:
            self.deadline = new_deadline

    def clear_deadline(self):
        self.deadline = None

    def past_deadline(self):
        return self.deadline and time.monotonic() > self.deadline

    # These mirror methods from GRPC Context

    def add_callback(self, callback):
        self.cancel_callback = callback

    def set_code(self, code):
        self.code = code

    def set_details(self, message):
        self.message = message

    def abort(self, code, details):
        if code == grpc.StatusCode.OK:
            raise ValueError("Abort called with OK as status code")

        self.set_code(code)
        self.set_details(details)

        raise grpc.RpcError()

    def invocation_metadata(self):
        return []


def rescale_mode_to_fit_and_pad(mode):
    # Calculate fit mode
    if mode == generation_pb2.RESCALE_STRICT:
        fit = "strict"
    elif mode == generation_pb2.RESCALE_COVER:
        fit = "cover"
    else:
        fit = "contain"

    # Calculate pad mode (should only be used for CONTAIN modes)
    pad_mode = "constant"
    if mode == generation_pb2.RESCALE_CONTAIN_REPLICATE:
        pad_mode = "replicate"
    elif mode == generation_pb2.RESCALE_CONTAIN_REFLECT:
        pad_mode = "reflect"

    return fit, pad_mode


def apply_image_adjustment(
    manager, tensor, adjustments, native_width=None, native_height=None
):
    if not adjustments:
        return tensor

    log_message = "{}"
    log_args = [tensor]

    # Background Removal mask is stored outside loop, in case we want to reuse it
    bgmask = None

    for adjustment in adjustments:
        which = adjustment.WhichOneof("adjustment")

        engine_id = None
        if adjustment.HasField("engine_id"):
            engine_id = adjustment.engine_id

        if which == "blur":
            sigma = adjustment.blur.sigma
            direction = adjustment.blur.direction

            if direction == generation_pb2.DIRECTION_DOWN:
                tensor = images.directionalblur(tensor, sigma, "down")
            elif direction == generation_pb2.DIRECTION_UP:
                tensor = images.directionalblur(tensor, sigma, "up")
            else:
                tensor = images.gaussianblur(tensor, sigma)

        elif which == "invert":
            tensor = images.invert(tensor)

        elif which == "levels":
            tensor = images.levels(
                tensor,
                adjustment.levels.input_low,
                adjustment.levels.input_high,
                adjustment.levels.output_low,
                adjustment.levels.output_high,
            )

        elif which == "channels":
            tensor = images.channelmap(
                tensor,
                [
                    adjustment.channels.r,
                    adjustment.channels.g,
                    adjustment.channels.b,
                    adjustment.channels.a,
                ],
            )

        elif which == "rescale" or which == "autoscale":
            if which == "autoscale":
                mode = adjustment.autoscale.mode

                width = height = None
                if adjustment.autoscale.HasField("width"):
                    width = adjustment.autoscale.width
                if adjustment.autoscale.HasField("height"):
                    height = adjustment.autoscale.height

                if width is None and height is None:
                    if native_width is None or native_height is None:
                        raise ValueError(
                            "Can't use a full autoscale - insufficiently bound width or height"
                        )

                    width = native_width
                    height = native_height
                elif width is None:
                    assert height is not None
                    width = height / tensor.shape[-2] * tensor.shape[-1]
                elif height is None:
                    assert width is not None
                    height = width / tensor.shape[-1] * tensor.shape[-2]

            else:
                mode = adjustment.rescale.mode
                width, height = adjustment.rescale.width, adjustment.rescale.height

            # Rescale only if needed
            if tensor.shape[-2] != height or tensor.shape[-1] != width:
                fit, pad_mode = rescale_mode_to_fit_and_pad(mode)
                tensor = images.rescale(tensor, height, width, fit, pad_mode)

        elif which == "crop":
            tensor = images.crop(
                tensor,
                adjustment.crop.top,
                adjustment.crop.left,
                adjustment.crop.height,
                adjustment.crop.width,
            )

        elif which == "depth":
            with manager.with_engine(engine_id, "depth") as estimator:
                tensor = estimator(tensor)

        elif which == "normal":
            # Defaults
            kwargs = dict(background_threshold=0, preblur=0, postblur=5, smoothing=0.8)

            # Update with fields from adjustment
            for f in kwargs.keys():
                if adjustment.normal.HasField(f):
                    kwargs[f] = getattr(adjustment.normal, f)

            # Handle auto-masking
            mask = None
            if kwargs["background_threshold"] < 0:
                kwargs["background_threshold"] = 0
                with manager.with_engine(task="background-removal") as remover:
                    mask = remover(tensor, mode="mask")

            # Normal calculation can use depth or normal estimator pipelines
            task = "normal"
            if engine_id:
                spec = manager._find_spec(id=engine_id)
                task = spec.task

            if task == "depth":
                with manager.with_engine(engine_id, "depth") as estimator:
                    # 16384 gives roughly same result as normalise=False for MiDaS, but also works with other models like Zoe
                    # A slightly lower number doesn't blow out result so much though.
                    tensor = estimator(tensor) * 2048  # * 16384

                tensor = images.normalmap_from_depthmap(tensor, mask=mask, **kwargs)

            elif task == "normal":
                with manager.with_engine(engine_id, "normal") as baenormal:
                    tensor = baenormal(tensor)

                if mask is not None:
                    tensor = torch.cat([tensor, mask], dim=1)

            else:
                raise ValueError(
                    f"Engine ID {engine_id} is for task '{task}' not normal or depth"
                )

        elif which == "canny_edge":
            tensor = images.canny_edge(
                tensor,
                adjustment.canny_edge.low_threshold,
                adjustment.canny_edge.high_threshold,
            )

        elif which == "edge_detection":
            with manager.with_engine(engine_id, "edge_detection") as detector:
                tensor = detector(tensor)

        elif which == "segmentation":
            with manager.with_engine(engine_id, "segmentation") as segmentor:
                tensor = segmentor(tensor)

        elif which == "keypose":
            with manager.with_engine(engine_id, "pose") as estimator:
                tensor = estimator(tensor, output_format="keypose")

        elif which == "openpose":
            with manager.with_engine(engine_id, "pose") as estimator:
                tensor = estimator(tensor, output_format="openpose")

        elif which == "background_removal":
            if adjustment.background_removal.reapply:
                if bgmask is None:
                    raise ValueError("No mask memorised to reapply")

            else:
                with manager.with_engine(engine_id, "background-removal") as remover:
                    # Calculate the mask
                    bgmask = remover(tensor, mode="mask")

            BRM = generation_pb2.BackgroundRemovalMode

            # First, read mode, setting default if field not provided
            if adjustment.background_removal.HasField("mode"):
                mode = adjustment.background_removal.mode
            else:
                mode = BRM.ALPHA

            # Then apply mode
            if mode == BRM.NOTHING:
                pass
            else:
                tensor = images.normalise_tensor(tensor, 3)
                if mode == BRM.ALPHA:
                    tensor = torch.cat([tensor, bgmask], dim=1)
                elif mode == BRM.BLUR:
                    bg = images.infill(tensor, bgmask, 26)
                    bg = images.gaussianblur(bg, 13)
                    tensor = tensor * bgmask + bg * (1 - bgmask)
                elif mode == BRM.SOLID:
                    tensor = tensor * bgmask
                else:
                    raise ValueError("Unknown background removal mode")

        elif which == "palletize":
            colours = 8
            if adjustment.palletize.HasField("colours"):
                colours = adjustment.palletize.colours

            tensor = images.palletize(tensor, colours)

        elif which == "quantize":
            thresholds = list(adjustment.quantize.threshold)
            tensor = images.quantize(tensor, thresholds)

        elif which == "shuffle":
            tensor = images.shuffle(tensor)

        else:
            raise ValueError(f"Unkown image adjustment {which}")

        log_message += " > {} > {}"
        log_args += [which, tensor]

    logger.debug(vr(log_message, *log_args))
    return tensor


class InvalidArgumentError(Exception):
    pass


class ParameterExtractor:
    """
    ParameterExtractor pulls fields out of a deeply nested GRPC structure.

    Every method that doesn't start with an "_" is a field that can be
    extracted from a Request object

    They shouldn't be called directly, but through "get", which will
    memo-ise the result.
    """

    def __init__(
        self,
        manager,
        request,
        tensor_cache,
        resource_provider,
        api_variant: Literal["standard", "stable_studio"] = "standard",
    ):
        self._manager = manager
        self._request = request
        self._tensor_cache = tensor_cache
        self._resource_provider = resource_provider
        self._api_variant = api_variant

        self._echo = None

        # Handle variations
        if self._api_variant == "stable_studio":
            for prompt in self._prompt_of_type("artifact"):
                if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                    prompt.artifact.adjustments.insert(
                        0,
                        generation_pb2.ImageAdjustment(
                            invert=generation_pb2.ImageAdjustment_Invert()
                        ),
                    )

        # Add a cache to self.get to prevent multiple requests from recalculating
        # self.get = functools.cache(self.get)

    def _cache_tensors(self, artifact, tensors, metadata=None):
        if not artifact.HasField("cache_control"):
            return

        self._tensor_cache.set(
            artifact.cache_control.cache_id,
            tensors=tensors,
            metadata=metadata,
            max_age=artifact.cache_control.max_age,
        )

    def _cache_set(self, artifact, safetensors):
        if not artifact.HasField("cache_control"):
            return

        self._tensor_cache.set_safetensors(
            artifact.cache_control.cache_id,
            safetensors,
            max_age=artifact.cache_control.max_age,
        )

    def _cache_get(self, cache_id):
        return self._tensor_cache.get_safetensors(cache_id)

    def _image_from_artifact_binary(self, artifact):
        # Handle webp artifacts
        if artifact.mime == "image/webp" or artifact.magic == "WEBP":
            if not images.SUPPORTS_WEBP:
                raise InvalidArgumentError("Server does not support webp")

            return images.fromWebpBytes(artifact.binary).to(self._manager.mode.device)

        # Check mime and magic to make sure it's not some unknown type
        elif artifact.mime and artifact.mime != "image/png":
            raise InvalidArgumentError(f"Unknown mime type {artifact.mime}")

        elif artifact.magic and artifact.magic != "PNG":
            raise InvalidArgumentError(f"Unknown magic code {artifact.magic}")

        # So assume it's a PNG (either no mime / magic, or they are for PNG)
        return images.fromPngBytes(artifact.binary).to(self._manager.mode.device)

    def _image_from_artifact_reference(self, artifact):
        if artifact.ref.WhichOneof("reference") == "id":
            test = lambda x: x.id == artifact.ref.id
        else:
            test = lambda x: x.uuid == artifact.ref.uuid

        for prompt in self._prompt_of_type("artifact"):
            if test(prompt.artifact):
                return self._image_from_artifact(prompt.artifact, artifact.ref.stage)

    def _image_from_artifact(
        self,
        artifact: generation_pb2.Artifact,
        stage=generation_pb2.ARTIFACT_AFTER_ADJUSTMENTS,
        extra=None,
    ):
        if artifact.WhichOneof("data") == "binary":
            image = self._image_from_artifact_binary(artifact)
        elif artifact.WhichOneof("data") == "ref":
            image = self._image_from_artifact_reference(artifact)
        else:
            raise ValueError(
                f"Can't convert Artifact of type {artifact.WhichOneof('data')} to an image"
            )

        kwargs = dict(manager=self._manager)

        artifact_is_init_image = artifact.type == generation_pb2.ARTIFACT_IMAGE
        kwargs["native_width"] = self.width(use_init_image=not artifact_is_init_image)
        kwargs["native_height"] = self.height(use_init_image=not artifact_is_init_image)

        if stage != generation_pb2.ARTIFACT_BEFORE_ADJUSTMENTS:
            image = apply_image_adjustment(
                tensor=image, adjustments=artifact.adjustments, **kwargs
            )
        if stage == generation_pb2.ARTIFACT_AFTER_POSTADJUSTMENTS:
            image = apply_image_adjustment(
                tensor=image, adjustments=artifact.postAdjustments, **kwargs
            )
        if extra is not None:
            extra = extra if isinstance(extra, list) else [extra]
            image = apply_image_adjustment(tensor=image, adjustments=extra, **kwargs)

        return image

    def _lora_from_artifact_cache(self, artifact: generation_pb2.Artifact):
        return self._cache_get(artifact.cache_id)

    def _lora_from_artifact_lora(self, artifact: generation_pb2.Artifact):
        logger.warn("artifact.lora is deprecated, use artifact.safetensors instead")
        return deserialize_safetensors(artifact.lora.lora)

    def _lora_from_artifact_safetensors(self, artifact: generation_pb2.Artifact):
        return deserialize_safetensors(artifact.safetensors)

    def _lora_from_artifact_url(self, artifact: generation_pb2.Artifact):
        return self._resource_provider.get("lora", artifact.url)

    def _lora_from_artifact(self, artifact: generation_pb2.Artifact):
        if artifact.WhichOneof("data") == "cache_id":
            lora = self._lora_from_artifact_cache(artifact)
        elif artifact.WhichOneof("data") == "lora":
            lora = self._lora_from_artifact_lora(artifact)
        elif artifact.WhichOneof("data") == "safetensors":
            lora = self._lora_from_artifact_safetensors(artifact)
        elif artifact.WhichOneof("data") == "url":
            lora = self._lora_from_artifact_url(artifact)
        else:
            raise ValueError(
                f"Can't convert Artifact of type {artifact.WhichOneof('data')} to an LoRA"
            )

        self._cache_set(artifact, lora)
        return lora

    def _token_embedding_from_artifact_cache(self, artifact: generation_pb2.Artifact):
        return self._cache_get(artifact.cache_id)

    def _token_embedding_from_artifact_te(self, artifact: generation_pb2.Artifact):
        logger.warn(
            "artifact.token_embedding is deprecated, use artifact.safetensors instead"
        )
        text = artifact.token_embedding.text
        tensor = deserialize_tensor(artifact.token_embedding.tensor)
        return UserSafetensors(tensors={text: tensor})

    def _token_embedding_from_artifact_safetensors(
        self, artifact: generation_pb2.Artifact
    ):
        return deserialize_safetensors(artifact.safetensors)

    def _token_embedding_from_artifact_url(self, artifact: generation_pb2.Artifact):
        return self._resource_provider.get("embedding", artifact.url)

    def _token_embedding_from_artifact(self, artifact: generation_pb2.Artifact):
        if artifact.WhichOneof("data") == "cache_id":
            embedding = self._token_embedding_from_artifact_cache(artifact)
        elif artifact.WhichOneof("data") == "token_embedding":
            embedding = self._token_embedding_from_artifact_te(artifact)
        elif artifact.WhichOneof("data") == "safetensors":
            embedding = self._token_embedding_from_artifact_safetensors(artifact)
        elif artifact.WhichOneof("data") == "url":
            embedding = self._token_embedding_from_artifact_url(artifact)
        else:
            raise ValueError(
                f"Can't convert Artifact of type {artifact.WhichOneof('data')} to an token embedding"
            )

        self._cache_set(artifact, embedding)
        return embedding

    def _image_stepparameter(self, field):
        if self._request.WhichOneof("params") != "image":
            return None

        for ctx in self._request.image.parameters:
            parts = field.split(".")

            while parts:
                if ctx.HasField(parts[0]):
                    ctx = getattr(ctx, parts.pop(0))
                else:
                    parts = ctx = None

            if ctx:
                return ctx

    def _image_parameter(self, field):
        if self._request.WhichOneof("params") != "image":
            return None
        if not self._request.image.HasField(field):
            return None
        return getattr(self._request.image, field)

    def _prompt_of_type(self, ptype) -> Iterable[generation_pb2.Prompt]:
        for prompt in self._request.prompt:
            which = prompt.WhichOneof("prompt")
            if which == ptype:
                yield prompt

    def _prompt_of_artifact_type(self, atype) -> Iterable[generation_pb2.Prompt]:
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == atype:
                yield prompt

    def _clip_layer_from_prompt(self, prompt):
        if prompt.HasField("parameters") and prompt.parameters.HasField("clip_layer"):
            return prompt.parameters.clip_layer

        return None

    def _prompt_by_weight(self, weight_callback=None):

        fragments = []
        clip_layer = None

        for prompt in self._prompt_of_type("text"):
            weight = 1.0
            if prompt.HasField("parameters") and prompt.parameters.HasField("weight"):
                weight = prompt.parameters.weight

            if weight_callback is not None:
                weight = weight_callback(weight)

            if weight > 0:
                fragments.append(PromptFragment(prompt=prompt.text, weight=weight))
                clip_layer = self._clip_layer_from_prompt(prompt)

        return Prompt(fragments=fragments, clip_layer=clip_layer) if fragments else None

    def prompt(self):
        return self._prompt_by_weight()

    def negative_prompt(self):
        return self._prompt_by_weight(lambda weight: -weight)

    def num_images_per_prompt(self):
        return self._image_parameter("samples")

    def height(self, use_init_image=True):
        # Just accept height if passed
        if (height := self._image_parameter("height")) is not None:
            return height
        # If not passed, first try and use the size from init image
        if use_init_image and (image := self.image) is not None:
            # If width parameter _was_ passed, calculate height to maintain aspect ratio
            if (width := self._image_parameter("width")) is not None:
                return round(image.shape[-2] / image.shape[-1] * width)
            # Otherwise just use image size
            return image.shape[-2]
        # Otherwise default to 512
        return 512

    def width(self, use_init_image=True):
        # Just accept width if passed
        if (width := self._image_parameter("width")) is not None:
            return width
        # If not passed, first try and use the size from init image
        if use_init_image and (image := self.image) is not None:
            # If height parameter _was_ passed, calculate width to maintain aspect ratio
            if (height := self._image_parameter("height")) is not None:
                return round(image.shape[-1] / image.shape[-2] * height)
            return image.shape[-1]
        # Otherwise default to 512
        return 512

    def seed(self):
        if self._request.WhichOneof("params") != "image":
            return None
        seed = list(self._request.image.seed)
        return seed if seed else None

    def guidance_scale(self):
        return self._image_stepparameter("sampler.cfg_scale")

    def clip_guidance_scale(self):
        if self._request.WhichOneof("params") != "image":
            return None

        for parameters in self._request.image.parameters:
            if parameters.HasField("guidance"):
                guidance = parameters.guidance
                for instance in guidance.instances:
                    if instance.HasField("guidance_strength"):
                        return instance.guidance_strength

    def sampler(self):
        if self._request.WhichOneof("params") != "image":
            return None
        if not self._request.image.HasField("transform"):
            return None
        if self._request.image.transform.WhichOneof("type") != "diffusion":
            return None
        return self._request.image.transform.diffusion

    def num_inference_steps(self):
        return self._image_parameter("steps")

    def eta(self):
        return self._image_stepparameter("sampler.eta")

    def churn(self):
        churn_settings = self._image_stepparameter("sampler.churn")
        return churn_settings.churn if churn_settings else None

    def churn_tmin(self):
        return self._image_stepparameter("sampler.churn.churn_tmin")

    def churn_tmax(self):
        return self._image_stepparameter("sampler.churn.churn_tmax")

    def sigma_min(self):
        return self._image_stepparameter("sampler.sigma.sigma_min")

    def sigma_max(self):
        return self._image_stepparameter("sampler.sigma.sigma_max")

    def karras_rho(self):
        return self._image_stepparameter("sampler.sigma.karras_rho")

    def scheduler_noise_type(self):
        noise_type = self._image_stepparameter("sampler.noise_type")

        if noise_type == generation_pb2.SAMPLER_NOISE_NORMAL:
            return "normal"
        if noise_type == generation_pb2.SAMPLER_NOISE_BROWNIAN:
            return "brownian"

        return None

    def _add_to_echo(self, prompt: generation_pb2.Prompt, image):
        if prompt.echo_back:
            if self._echo is None:
                answer = generation_pb2.Answer()
                answer.request_id = self._request.request_id
                answer.answer_id = "echo"
                self._echo = answer

            artifact = image_to_artifact(
                image, artifact_type=prompt.artifact.type, accept=self._request.accept
            )
            artifact.index = len(self._echo.artifacts) + 1
            self._echo.artifacts.append(artifact)

        return image

    @cached_property
    def image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_IMAGE:
                return self._add_to_echo(
                    prompt,
                    self._image_from_artifact(prompt.artifact),
                )

    def mask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                return self._add_to_echo(
                    prompt, self._image_from_artifact(prompt.artifact)
                )

    def outmask_image(self):
        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type == generation_pb2.ARTIFACT_MASK:
                return self._image_from_artifact(
                    prompt.artifact, generation_pb2.ARTIFACT_AFTER_POSTADJUSTMENTS
                )

    def hint_images(self):
        hint_images = []

        for prompt in self._prompt_of_type("artifact"):
            if prompt.artifact.type in {
                generation_pb2.ARTIFACT_HINT_IMAGE,
                generation_pb2.ARTIFACT_DEPTH,
            }:
                # Calculate weight
                weight = 1.0
                if prompt.HasField("parameters"):
                    if prompt.parameters.HasField("weight"):
                        weight = prompt.parameters.weight

                # Build the actual image
                hint_image = self._add_to_echo(
                    prompt,
                    self._image_from_artifact(prompt.artifact),
                )

                # Find the hint type (to handle deprecated ARTIFACT_DEPTH type)
                if prompt.artifact.type == generation_pb2.ARTIFACT_DEPTH:
                    hint_type = "depth"
                else:
                    hint_type = prompt.artifact.hint_image_type

                priority = "balanced"
                if prompt.parameters.HasField("hint_priority"):
                    priority_table: dict[Any, HintPriority] = {
                        generation_pb2.HINT_BALANCED: "balanced",
                        generation_pb2.HINT_PRIORITISE_HINT: "hint",
                        generation_pb2.HINT_PRIORITISE_PROMPT: "prompt",
                    }

                    priority = priority_table[prompt.parameters.hint_priority]

                # And append the details
                hint_images.append(
                    HintImage(
                        image=hint_image,
                        hint_type=hint_type,
                        weight=weight,
                        priority=priority,
                        clip_layer=self._clip_layer_from_prompt(prompt),
                    )
                )

        return hint_images

    def lora(self):
        loras = []

        for prompt in self._prompt_of_artifact_type(generation_pb2.ARTIFACT_LORA):
            safetensors = self._lora_from_artifact(prompt.artifact)
            weights = {}

            if prompt.HasField("parameters"):
                if prompt.parameters.HasField("weight"):
                    weights["*"] = prompt.parameters.weight
                for named_weight in prompt.parameters.named_weights:
                    weights[named_weight.name] = named_weight.weight

            loras.append((safetensors, weights))

        return loras if loras else None

    def token_embeddings(self):
        embeddings = {}

        for prompt in self._prompt_of_artifact_type(
            generation_pb2.ARTIFACT_TOKEN_EMBEDDING
        ):
            embedding = self._token_embedding_from_artifact(prompt.artifact)
            tensors = embedding.tensors()

            if prompt.HasField("parameters"):
                free_overrides = []
                named_overrides = {}

                for override in prompt.parameters.token_overrides:
                    if override.HasField("original_token"):
                        named_overrides[override.original_token] = override.token
                    else:
                        free_overrides.append(override.token)

                if free_overrides or named_overrides:
                    tensors = {}
                    for key, tensor in embedding.items():
                        if key in named_overrides:
                            tensors[named_overrides[key]] = tensor
                        elif free_overrides:
                            tensors[free_overrides.pop(0)] = tensor
                        else:
                            tensors[key] = tensor

            embeddings.update(tensors)

        return embeddings

    def strength(self):
        return self._image_stepparameter("schedule.start")

    def hires_fix(self):
        hires = self._image_parameter("hires")
        return hires.enable if hires else None

    def hires_oos_fraction(self):
        hires = self._image_parameter("hires")
        if hires and hires.HasField("oos_fraction"):
            return hires.oos_fraction
        return None

    def tiling(self):
        tiling: set[str] = set()

        # Handle tiling initially setting the status, but then the
        # two specific axis being able to override if present
        if self._image_parameter("tiling") is True:
            tiling = {"x", "y"}

        if self._image_parameter("tiling_x") is True:
            tiling |= {"x"}
        elif self._image_parameter("tiling_x") is False:
            tiling -= {"x"}

        if self._image_parameter("tiling_y") is True:
            tiling |= {"y"}
        elif self._image_parameter("tiling_y") is False:
            tiling -= {"y"}

        # Turning a set into a string isn't predictable in ordering
        if len(tiling) == 2:
            return True
        elif tiling:
            return list(tiling)[0]
        else:
            return False

    def get(self, field):
        val = getattr(self, field)
        return val() if callable(val) else val

    def fields(self):
        return [
            key
            for key in dir(self)
            if key[0] != "_" and key != "get" and key != "fields"
        ]


class GenerationServiceServicer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(
        self,
        manager,
        tensor_cache,
        resource_provider,
        supress_metadata=False,
        debug_recorder=DebugNullRecorder(),
        ram_monitor=None,
    ):
        self._manager = manager
        self._tensor_cache = tensor_cache
        self._resource_provider = resource_provider
        self._supress_metadata = supress_metadata
        self._debug_recorder = debug_recorder
        self._ram_monitor = ram_monitor

        # For async support
        self._async_contexts_lock = threading.Lock()
        self._async_contexts: dict[str, AsyncContext] = {}

    def unimp(self, what):
        raise NotImplementedError(f"{what} not implemented")

    def batched_seeds(self, samples, seeds, batchmax):
        # If we weren't given any seeds at all, just start with a single -1
        if not seeds:
            seeds = [-1]

        # Replace any negative seeds with a randomly selected one
        seeds = [
            seed if seed > 0 else random.randrange(0, 2**32 - 1) for seed in seeds
        ]

        # Fill seeds up to params.samples if we didn't get passed enough
        if len(seeds) < samples:
            # Starting with the last seed we were given
            nextseed = seeds[-1] + 1
            while len(seeds) < samples:
                seeds.append(nextseed)
                nextseed += 1

        # Calculate the most even possible split across batchmax
        if samples <= batchmax:
            batches = [samples]
        elif samples % batchmax == 0:
            batches = [batchmax] * (samples // batchmax)
        else:
            d = samples // batchmax + 1
            batchsize = samples // d
            r = samples - batchsize * d
            batches = [batchsize + 1] * r + [batchsize] * (d - r)

        for batch in batches:
            batchseeds, seeds = seeds[:batch], seeds[batch:]
            yield batchseeds

    def generate_request(
        self,
        request,
        api_variant: Literal["standard", "stable_studio"],
        stop_event,
        recorder,
    ):

        # Assume that "None" actually means "Image" (stability-sdk/client.py doesn't set it)
        if (
            request.requested_type != generation_pb2.ARTIFACT_NONE
            and request.requested_type != generation_pb2.ARTIFACT_IMAGE
        ):
            self.unimp("Generation of anything except images")

        extractor = ParameterExtractor(
            request=request,
            manager=self._manager,
            tensor_cache=self._tensor_cache,
            resource_provider=self._resource_provider,
            api_variant=api_variant,
        )
        kwargs = {}

        for field in extractor.fields():
            val = extractor.get(field)
            if val is not None:
                kwargs[field] = val

        if extractor._echo is not None:
            yield extractor._echo

        if self._ram_monitor:
            print("Arguments processed")
            self._ram_monitor.print()

        if request.engine_id == "noop":
            image = kwargs.get("image", None)

            # If there was an init image passed, return the result
            if image is not None:
                answer = generation_pb2.Answer()
                answer.request_id = request.request_id
                answer.answer_id = "noop"
                artifact = image_to_artifact(image, accept=request.accept)
                artifact.index = 1
                artifact.uuid = str(uuid.uuid4())
                answer.artifacts.append(artifact)
                yield answer

            return

        ctr = 0
        samples = kwargs.get("num_images_per_prompt", 1)
        seeds = kwargs.get("seed", None)
        batchmax = self._manager.batchMode.batchmax(kwargs["width"] * kwargs["height"])

        for seeds in self.batched_seeds(samples, seeds, batchmax):
            batchargs = {
                **kwargs,
                "seed": seeds,
                "num_images_per_prompt": len(seeds),
            }

            logargs = {**batchargs}
            if "lora" in logargs:
                value = logargs["lora"]
                logargs["lora"] = (
                    f"[{len(value)}]" if isinstance(value, list) else "yes"
                )

            if "token_embeddings" in logargs:
                logargs["token_embeddings"] = list(logargs["token_embeddings"].keys())

            logstr = []
            loglate = []
            for k, v in logargs.items():
                if k == "hint_images" and v:
                    for i, hint_image in enumerate(v):
                        loglate.append(
                            f"hint_image: {{hint_images[{i}].image}} ("
                            f"hint_type: {{hint_images[{i}].hint_type}} "
                            f"weight: {{hint_images[{i}].weight:.2f}} "
                            f"priority: {{hint_images[{i}].priority}} "
                            f"clip_layer: {{hint_images[{i}].clip_layer}}"
                            f")"
                        )
                elif k in {"image", "mask_image", "outmask_image"}:
                    loglate.append(f"{k}: {{{k}}}")
                else:
                    logstr.append(f"{k}: {{{k}}}")

            logger.info("Generating:")
            for line in logstr + loglate:
                logger.info(vr("  " + line, **logargs))

            recorder.store("pipe.generate calls", kwargs)

            with self._manager.with_engine(request.engine_id) as engine:
                results = engine(**batchargs, stop_event=stop_event)

            meta = pb_json_format.MessageToDict(request)
            binary_fields = [
                "binary",
                "tokens",
                "tensor",
                "safetensors",
                "lora",
                "token_embedding",
            ]
            for prompt in meta["prompt"]:
                if "artifact" in prompt:
                    for field in binary_fields:
                        if field in prompt["artifact"]:
                            prompt["artifact"][field] = "-binary-"

            if results is None:
                result_images, nsfws = [], []
            elif isinstance(results, torch.Tensor):
                result_images = results
                nsfws = [False] * len(result_images)
            elif len(results) == 1:
                result_images = results[0]
                nsfws = [False] * len(result_images)
            else:
                result_images, nsfws = results[0], results[1]

            for i, (result_image, nsfw) in enumerate(zip(result_images, nsfws)):
                answer = generation_pb2.Answer()
                answer.request_id = request.request_id
                answer.answer_id = f"{request.request_id}-{ctr}"

                img_seed = seeds[i] if i < len(seeds) else 0

                if self._supress_metadata:
                    artifact = image_to_artifact(result_image, accept=request.accept)
                else:
                    meta.setdefault("image", {})["samples"] = 1
                    meta.setdefault("image", {})["seed"] = [img_seed]
                    artifact = image_to_artifact(
                        result_image,
                        accept=request.accept,
                        encoded_parameters=json.dumps(meta),
                    )

                artifact.finish_reason = (
                    generation_pb2.FILTER if nsfw else generation_pb2.NULL
                )
                artifact.index = ctr
                artifact.seed = img_seed
                artifact.uuid = str(uuid.uuid4())
                answer.artifacts.append(artifact)

                logger.debug(vr("Result {image}", image=result_image))
                recorder.store("pipe.generate result", artifact)

                yield answer
                ctr += 1

        if self._ram_monitor:
            self._ram_monitor.print()

    def api_variant_from_context(self, context):
        # Detect stable studio, which sends mask inverted compared to previous standard
        for key, value in context.invocation_metadata():
            if key == "stability-client-id" and value == "StableStudio":
                return "stable_studio"

        return "standard"

    @exception_to_grpc(
        {
            EngineNotFoundError: grpc.StatusCode.NOT_FOUND,
            NotImplementedError: grpc.StatusCode.UNIMPLEMENTED,
            InvalidArgumentError: grpc.StatusCode.INVALID_ARGUMENT,
            CacheLookupError: lambda e, d: (
                grpc.StatusCode.FAILED_PRECONDITION,
                e.args[0],
                f"Cache miss, key {e.args[0]}",
            ),
        }
    )
    def Generate(self, request, context):
        with self._debug_recorder.record(request.request_id) as recorder:
            recorder.store("generate request", request)

            api_variant = self.api_variant_from_context(context)
            stop_event = threading.Event()
            context.add_callback(lambda: stop_event.set())

            for answer in self.generate_request(
                request, api_variant, stop_event, recorder
            ):
                yield answer

                if stop_event.is_set():
                    break

    @exception_to_grpc(
        {
            EngineNotFoundError: grpc.StatusCode.NOT_FOUND,
            NotImplementedError: grpc.StatusCode.UNIMPLEMENTED,
            InvalidArgumentError: grpc.StatusCode.INVALID_ARGUMENT,
            CacheLookupError: lambda e, d: (
                grpc.StatusCode.FAILED_PRECONDITION,
                e.args[0],
                f"Cache miss, key {e.args[0]}",
            ),
        }
    )
    def ChainGenerate(self, request: generation_pb2.ChainRequest, context):
        class StatusHandler:
            def __init__(self, handler):
                self.reason = {r for r in handler.reason} if handler.reason else {0}
                self.action = {a for a in handler.action} if handler.action else {0}
                self.target = handler.target if handler.HasField("target") else None

        with self._debug_recorder.record(request.request_id) as recorder:
            recorder.store("generate chain request", request)

            api_variant = self.api_variant_from_context(context)
            stop_event = threading.Event()
            context.add_callback(lambda: stop_event.set())

            input_artifacts = {}

            for stage in request.stage:
                if stage.request.engine_id == "asset-service":
                    logger.warn("assert-service not currently implemented. Skipping.")
                    continue

                # Pull out the on_status list to be matchable more easily
                status_handlers = [
                    StatusHandler(handler) for handler in stage.on_status
                ]

                builders = [lambda x: None]

                # TODO: It's not clear what should be done for subsequent targets in the
                # chain when there are multiple input images. For now, we split it into
                # multiple generations on the target.
                if stage.id in input_artifacts:
                    builders = [
                        lambda stage_request: stage_request.prompt.append(artifact)
                        for artifact in input_artifacts[stage.id]
                    ]

                for builder in builders:
                    stage_request = generation_pb2.Request()
                    stage_request.CopyFrom(stage.request)
                    stage_request.request_id = stage.id
                    builder(stage_request)

                    recorder.store("generate chain stage request", stage_request)

                    for answer in self.generate_request(
                        stage_request, api_variant, stop_event, recorder
                    ):
                        assert len(answer.artifacts) == 1

                        artifact = answer.artifacts[0]
                        for handler in status_handlers:
                            if artifact.finish_reason not in handler.reason:
                                continue

                            # Handle RETURN action
                            if generation_pb2.STAGE_ACTION_RETURN in handler.action:
                                yield answer

                            # Handle PASS action
                            if generation_pb2.STAGE_ACTION_PASS in handler.action:
                                store = input_artifacts.setdefault(handler.target, [])
                                store.append(artifact)

                        if stop_event.is_set():
                            break

    def _try_deleting_context(self, key):
        """
        Since multiple threads might be deleting contexts, we need to wrap
        it in a try block to avoid failing if we attempt to double-delete.
        """
        try:
            del self._async_contexts[key]
        except KeyError:
            pass

    def _check_deadlines(self):
        deadline_expired = [
            key for key, value in self._async_contexts.items() if value.past_deadline()
        ]

        for key in deadline_expired:
            self._try_deleting_context(key)

    @exception_to_grpc
    def AsyncGenerate(self, request: generation_pb2.Request, context):
        self._check_deadlines()

        async_context = AsyncContext()
        check_context = None
        handle = None

        # Find an unusued handle.
        while check_context is not async_context:
            handle = str(uuid.uuid4())
            # Done a slightly weird way to ensure dict access is atomic
            check_context = self._async_contexts.setdefault(handle, async_context)

        # Start the request in a thread.
        # TODO: Ideally this would be in a queue too rather than spawning threads

        def thread_function():
            try:
                for answer in self.Generate(request, async_context):
                    async_context.queue.put(answer)
            except grpc.RpcError:
                # RpcError will have set code and details already in context#abort
                pass
            finally:
                async_context.queue.put("DONE")
                # Remove queue after 10 minutes, to avoid queues that never get
                # emptied by clients from consuming memory
                async_context.set_deadline(60 * 10)

        async_context.thread = threading.Thread(target=thread_function)
        async_context.thread.start()

        return generation_pb2.AsyncHandle(
            request_id=request.request_id, async_handle=handle
        )

    @exception_to_grpc
    def AsyncResult(self, request, context):
        self._check_deadlines()

        async_context = self._async_contexts.get(request.async_handle)

        if not async_context:
            context.abort(grpc.StatusCode.NOT_FOUND, "No such async handle")
        assert async_context  # context.abort will raise an exception

        async_answer = generation_pb2.AsyncAnswer(complete=False)
        wait_args = dict(block=True, timeout=0.5)

        try:
            while True:
                answer = async_context.queue.get(**wait_args)

                if answer == "DONE":
                    async_answer.complete = True
                    async_answer.status.code = async_context.code.value[0]
                    async_answer.status.message = async_context.message
                    break

                async_answer.answer.append(answer)
                wait_args = dict(block=False, timeout=0)
        except Empty:
            pass

        if async_answer.complete:
            self._try_deleting_context(request.async_handle)

        return async_answer

    @exception_to_grpc
    def AsyncCancel(self, request, context):
        self._check_deadlines()

        async_context = self._async_contexts.get(request.async_handle)

        if not async_context:
            context.abort(grpc.StatusCode.NOT_FOUND, "No such async handle")
        assert async_context  # context.abort will raise an exception

        async_context.cancel()

        self._try_deleting_context(request.async_handle)

        return generation_pb2.AsyncCancelAnswer()
