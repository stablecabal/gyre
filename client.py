#!/bin/which python3

# Modified version of Stability-AI SDK client.py. Changes:
#   - Calls cancel on ctrl-c to allow server to abort
#   - Supports setting ETA parameter
#   - Supports actually setting CLIP guidance strength
#   - Supports negative prompt by setting a prompt with negative weight
#   - Supports sending key to machines on local network over HTTP (not HTTPS)

import hashlib
import io
import json
import logging
import mimetypes
import os
import pathlib
import random
import signal
import sys
import time
import uuid
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from itertools import zip_longest
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

import grpc
import machineid
import torch
import yaml
from google.protobuf.json_format import MessageToDict, MessageToJson
from PIL import Image, ImageOps
from safetensors.torch import safe_open

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    pass
else:
    load_dotenv()

# this is necessary because of how the auto-generated code constructs its imports
thisPath = pathlib.Path(__file__).parent.resolve()
genPath = thisPath / "gyre/generated"
sys.path.append(str(genPath))

import engines_pb2 as engines
import engines_pb2_grpc as engines_grpc
import generation_pb2 as generation
import generation_pb2_grpc as generation_grpc
import tensors_pb2 as tensors

from gyre.protobuf_safetensors import (
    serialize_safetensor,
    serialize_safetensor_from_dict,
)
from gyre.protobuf_tensors import serialize_tensor

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

SAMPLERS: Dict[str, int] = {
    "ddim": generation.SAMPLER_DDIM,
    "plms": generation.SAMPLER_DDPM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation.SAMPLER_K_LMS,
    "dpm_fast": generation.SAMPLER_DPM_FAST,
    "dpm_adaptive": generation.SAMPLER_DPM_ADAPTIVE,
    "dpmspp_1": generation.SAMPLER_DPMSOLVERPP_1ORDER,
    "dpmspp_2": generation.SAMPLER_DPMSOLVERPP_2ORDER,
    "dpmspp_3": generation.SAMPLER_DPMSOLVERPP_3ORDER,
    "dpmspp_2s_ancestral": generation.SAMPLER_DPMSOLVERPP_2S_ANCESTRAL,
    "dpmspp_sde": generation.SAMPLER_DPMSOLVERPP_SDE,
    "dpmspp_2m": generation.SAMPLER_DPMSOLVERPP_2M,
}

NOISE_TYPES: Dict[str, int] = {
    "normal": generation.SAMPLER_NOISE_NORMAL,
    "brownian": generation.SAMPLER_NOISE_BROWNIAN,
}


class GrpcAsyncError(Exception):
    def __init__(self, code, message):
        super().__init__()

        for possible in grpc.StatusCode:
            if possible.value[0] == code:
                self._code = possible
                break
        else:
            self._code = grpc.StatusCode.UNKNOWN

        self._message = message

    def code(self):
        return self._code

    def message(self):
        return self._message


def floatlike(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.

    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    algorithm = SAMPLERS.get(algorithm_key, None)
    if algorithm is None:
        raise ValueError(f"unknown sampler {s}")

    return algorithm


def get_noise_type_from_str(s: str) -> generation.SamplerNoiseType:
    noise_key = s.lower().strip()
    noise_type = NOISE_TYPES.get(noise_key, None)

    if noise_type is None:
        raise ValueError(f"unknown noise type {s}")

    return noise_type


def open_images(
    images: Union[
        Sequence[Tuple[str, generation.Artifact]],
        Generator[Tuple[str, generation.Artifact], None, None],
    ],
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Open the images from the filenames and Artifacts tuples.

    :param images: The tuples of Artifacts and associated images to open.
    :return:  A Generator of tuples of image filenames and Artifacts, intended
     for passthrough.
    """
    from PIL import Image

    for path, artifact in images:
        if artifact.type == generation.ARTIFACT_IMAGE:
            if verbose:
                logger.info(f"opening {path}")
            img = Image.open(io.BytesIO(artifact.binary))
            img.show()
        yield [path, artifact]


def image_to_prompt(
    im, init: bool = False, mask: bool = False, use_alpha=False
) -> generation.Prompt:
    if init and mask:
        raise ValueError("init and mask cannot both be True")

    if use_alpha:
        # Split into 3 channels
        r, g, b, a = im.split()
        # Recombine back to RGB image
        im = Image.merge("RGB", (a, a, a))
        im = ImageOps.invert(im)

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)

    artifact_uuid = str(uuid.uuid4())

    type = generation.ARTIFACT_IMAGE
    if mask:
        type = generation.ARTIFACT_MASK

    prompt = generation.Prompt(
        artifact=generation.Artifact(
            type=type, uuid=artifact_uuid, binary=buf.getvalue()
        ),
        parameters=generation.PromptParameters(init=init),
    )

    # bgremove = generation.ImageAdjustment(
    #     background_removal=generation.ImageAdjustment_BackgroundRemoval(
    #         mode=generation.BackgroundRemovalMode.BLUR
    #     )
    # )
    # prompt.artifact.adjustments.append(bgremove)

    return prompt


def add_converter_to_hint_image_prompt(prompt, remove_bg, converter, args):
    if converter is None or converter is False:
        return

    if remove_bg:
        bgremove = generation.ImageAdjustment(
            background_removal=generation.ImageAdjustment_BackgroundRemoval(
                mode=generation.BackgroundRemovalMode.SOLID
            )
        )
        prompt.artifact.adjustments.append(bgremove)

    adjustment = None
    hint_type = prompt.artifact.hint_image_type

    if "depth" in hint_type:
        adjustment = generation.ImageAdjustment(
            depth=generation.ImageAdjustment_Depth()
        )
    elif "canny" in hint_type:
        args = {"low_threshold": 100, "high_threshold": 200, **args}

        adjustment = generation.ImageAdjustment(
            canny_edge=generation.ImageAdjustment_CannyEdge(**args)
        )
    elif "hed" in hint_type or "softedge" in hint_type or "lineart" in hint_type:
        adjustment = generation.ImageAdjustment(
            edge_detection=generation.ImageAdjustment_EdgeDetection()
        )
    elif "sketch" in hint_type or "scribble" in hint_type:
        adjustment = [
            generation.ImageAdjustment(
                edge_detection=generation.ImageAdjustment_EdgeDetection()
            ),
            generation.ImageAdjustment(
                blur=generation.ImageAdjustment_Gaussian(sigma=3)
            ),
            generation.ImageAdjustment(
                quantize=generation.ImageAdjustment_Quantize(threshold=[0.15])
            ),
        ]
    elif "segment" in hint_type:
        adjustment = generation.ImageAdjustment(
            segmentation=generation.ImageAdjustment_Segmentation()
        )
    elif "keypose" in hint_type:
        adjustment = generation.ImageAdjustment(
            keypose=generation.ImageAdjustment_Keypose()
        )
    elif "openpose" in hint_type:
        adjustment = generation.ImageAdjustment(
            openpose=generation.ImageAdjustment_Openpose()
        )
    elif "normal" in hint_type:
        args = {"preblur": 0, "postblur": 0, **args}

        adjustment = generation.ImageAdjustment(
            normal=generation.ImageAdjustment_Normal(**args)
        )
    elif "color" in hint_type:
        args = {"colours": 8, **args}

        adjustment = generation.ImageAdjustment(
            palletize=generation.ImageAdjustment_Palletize(**args)
        )
    elif "shuffle" in hint_type:
        adjustment = [
            generation.ImageAdjustment(
                autoscale=generation.ImageAdjustment_Autoscale(
                    mode=generation.RESCALE_COVER
                )
            ),
            generation.ImageAdjustment(shuffle=generation.ImageAdjustment_Shuffle()),
        ]
    else:
        raise ValueError(f"Gyre can't convert image to hint type {hint_type}")

    if isinstance(adjustment, list):
        prompt.artifact.adjustments.extend(adjustment)

    else:
        if isinstance(converter, str):
            adjustment.engine_id = converter

        prompt.artifact.adjustments.append(adjustment)

    if remove_bg:
        bgremove = generation.ImageAdjustment(
            background_removal=generation.ImageAdjustment_BackgroundRemoval(
                mode=generation.BackgroundRemovalMode.ALPHA, reapply=True
            )
        )
        prompt.artifact.adjustments.append(bgremove)

    return prompt


def hint_image_to_prompt(
    image,
    hint_type,
    weight=1.0,
    priority=generation.HINT_BALANCED,
    remove_bg=False,
    converter=None,
    args={},
) -> generation.Prompt:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    artifact_uuid = str(uuid.uuid4())

    prompt = generation.Prompt(
        # echo_back=converter is not None,
        artifact=generation.Artifact(
            type=generation.ARTIFACT_HINT_IMAGE,
            uuid=artifact_uuid,
            binary=buf.getvalue(),
            hint_image_type=hint_type,
        ),
        parameters=generation.PromptParameters(weight=weight, hint_priority=priority),
    )

    add_converter_to_hint_image_prompt(prompt, remove_bg, converter, args)

    return prompt


def ref_to_prompt(ref_uuid, type, stage=generation.ARTIFACT_AFTER_ADJUSTMENTS):
    return generation.Prompt(
        artifact=generation.Artifact(
            type=type,
            ref=generation.ArtifactReference(uuid=ref_uuid, stage=stage),
        )
    )


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def cache_id(path):
    system_id = machineid.hashed_id("gyre-client")
    file_hash = sha256sum(path)
    return f"{system_id}-{file_hash}"


USE_DEPRECATED = False


def lora_to_prompt(path, weights, from_cache=False):
    parameters = generation.PromptParameters()

    if weights and len(weights) == 1:
        parameters.weight = weights[0]
    elif weights and len(weights) == 2:
        parameters.named_weights.append(
            generation.NamedWeight(name="unet", weight=weights[0])
        )
        parameters.named_weights.append(
            generation.NamedWeight(name="text_encoder", weight=weights[1])
        )

    if ":" in path:
        return generation.Prompt(
            artifact=generation.Artifact(type=generation.ARTIFACT_LORA, url=path),
            parameters=parameters,
        )

    elif from_cache:
        return generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_LORA, cache_id=cache_id(path)
            ),
            parameters=parameters,
        )

    else:
        ext = os.path.splitext(path)[1]

        if ext in {".bin", ".pt"}:
            tensordict = torch.load(path, "cpu")
            grpc_safetensors = serialize_safetensor_from_dict(tensordict)
        else:
            safetensors = safe_open(path, framework="pt", device="cpu")
            grpc_safetensors = serialize_safetensor(safetensors)

        return generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_LORA,
                safetensors=grpc_safetensors,
                cache_control=generation.CacheControl(
                    cache_id=cache_id(path), max_age=60 * 60  # Cache for an hour
                ),
            ),
            parameters=parameters,
        )


def ti_to_prompts(path, override_tokens, from_cache=False):

    parameters = generation.PromptParameters()
    for token in override_tokens:
        parameters.token_overrides.append(generation.TokenOverride(token=token))

    if ":" in path:
        prompt = generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_TOKEN_EMBEDDING, url=path
            ),
            parameters=parameters,
        )

    elif from_cache:
        prompt = generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_TOKEN_EMBEDDING, cache_id=cache_id(path)
            ),
            parameters=parameters,
        )

    else:
        data = torch.load(path, "cpu")

        if "string_to_param" in data:
            data = data["string_to_param"]

        prompt = generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_TOKEN_EMBEDDING,
                safetensors=serialize_safetensor_from_dict(data),
                cache_control=generation.CacheControl(
                    cache_id=cache_id(path), max_age=60 * 60  # Cache for an hour
                ),
            ),
            parameters=parameters,
        )

    return [prompt]


def process_artifacts_from_answers(
    prefix: str,
    answers: Union[
        Generator[generation.Answer, None, None], Sequence[generation.Answer]
    ],
    write: bool = True,
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Process the Artifacts from the Answers.

    :param prefix: The prefix for the artifact filenames.
    :param answers: The Answers to process.
    :param write: Whether to write the artifacts to disk.
    :param verbose: Whether to print the artifact filenames.
    :return: A Generator of tuples of artifact filenames and Artifacts, intended
        for passthrough.
    """
    idx = 0
    for resp in answers:
        for artifact in resp.artifacts:
            artifact_p = f"{prefix}-{resp.request_id}-{resp.answer_id}-{idx}"
            if artifact.type in {
                generation.ARTIFACT_IMAGE,
                generation.ARTIFACT_MASK,
                generation.ARTIFACT_HINT_IMAGE,
            }:
                if artifact.mime == "image/webp":
                    ext = ".webp"
                else:
                    ext = mimetypes.guess_extension(artifact.mime)
                contents = artifact.binary
            elif artifact.type == generation.ARTIFACT_CLASSIFICATIONS:
                ext = ".pb.json"
                contents = MessageToJson(artifact.classifier).encode("utf-8")
            elif artifact.type == generation.ARTIFACT_TEXT:
                ext = ".pb.json"
                contents = MessageToJson(artifact).encode("utf-8")
            else:
                ext = ".pb"
                contents = artifact.SerializeToString()
            out_p = f"{artifact_p}{ext}"
            if write:
                with open(out_p, "wb") as f:
                    f.write(bytes(contents))
                    if verbose:
                        artifact_t = generation.ArtifactType.Name(artifact.type)
                        logger.info(f"wrote {artifact_t} to {out_p}")
                        if artifact.finish_reason == generation.FILTER:
                            logger.info(f"{artifact_t} flagged as NSFW")

            yield [out_p, artifact]
            idx += 1


class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = "",
        proto: Literal["grpc", "grpc-web"] = "grpc",
        engine: str = "stable-diffusion-v1-5",
        verbose: bool = False,
        wait_for_ready: bool = True,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param verbose: Whether to print debug messages.
        :param wait_for_ready: Whether to wait for the server to be ready, or
            to fail immediately.
        """
        self.verbose = verbose
        self.engine = engine

        self.grpc_args = {}
        if proto == "grpc":
            self.grpc_args["wait_for_ready"] = wait_for_ready

        if verbose:
            logger.info(f"Opening channel to {host}")

        maxMsgLength = 256 * 1024 * 1024  # 256 MB

        channel_options = [
            ("grpc.max_message_length", maxMsgLength),
            ("grpc.max_send_message_length", maxMsgLength),
            ("grpc.max_receive_message_length", maxMsgLength),
        ]

        call_credentials = []

        if proto == "grpc-web":
            from gyre.sonora import client as sonora_client

            channel = sonora_client.insecure_web_channel(host)
        elif key:
            call_credentials.append(grpc.access_token_call_credentials(f"{key}"))

            if host.endswith("443"):
                channel_credentials = grpc.ssl_channel_credentials()
            else:
                print(
                    "Key provided but channel is not HTTPS - assuming a local network"
                )
                channel_credentials = grpc.local_channel_credentials()

            channel = grpc.secure_channel(
                host,
                grpc.composite_channel_credentials(
                    channel_credentials, *call_credentials
                ),
                options=channel_options,
            )
        else:
            channel = grpc.insecure_channel(host, options=channel_options)

        if verbose:
            logger.info(f"Channel opened to {host}")
        self.stub = generation_grpc.GenerationServiceStub(channel)
        self.engine_stub = engines_grpc.EnginesServiceStub(channel)

    def list_engines(self, task_group=engines.GENERATE):
        request = engines.ListEnginesRequest(task_group=task_group)
        print(self.engine_stub.ListEngines(request))

    def generate(
        self,
        prompt: Union[str, List[str], generation.Prompt, List[generation.Prompt]],
        negative_prompt: str = None,
        clip_layer: Optional[int] = None,
        init_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        mask_from_image_alpha: bool = False,
        height: int | None = None,
        width: int | None = None,
        start_schedule: float = 1.0,
        end_schedule: float = 0.01,
        cfg_scale: float = 7.0,
        eta: float = 0.0,
        churn: float = None,
        churn_tmin: float = None,
        churn_tmax: float = None,
        sigma_min: float = None,
        sigma_max: float = None,
        karras_rho: float = None,
        noise_type: int = None,
        sampler: generation.DiffusionSampler = generation.SAMPLER_K_LMS,
        steps: int = 50,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        safety: bool = True,
        classifiers: Optional[generation.ClassifierParameters] = None,
        guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
        guidance_cuts: int = 0,
        guidance_strength: Optional[float] = None,
        guidance_prompt: Union[str, generation.Prompt] = None,
        guidance_models: List[str] = None,
        hires_fix: bool | None = None,
        hires_oos_fraction: float | None = None,
        tiling: str = "no",
        hint_images: list[dict[str, str | float]] | None = None,
        lora: list[tuple[str, list[float]]] | None = None,
        ti: list[tuple[str, list[str]]] | None = None,
        as_async=False,
        from_cache=True,
        accept_webp=True,
    ) -> Generator[generation.Answer, None, None]:
        """
        Generate images from a prompt.

        :param prompt: Prompt to generate images from.
        :param init_image: Init image.
        :param mask_image: Mask image
        :param height: Height of the generated images.
        :param width: Width of the generated images.
        :param start_schedule: Start schedule for init image.
        :param end_schedule: End schedule for init image.
        :param cfg_scale: Scale of the configuration.
        :param sampler: Sampler to use.
        :param steps: Number of steps to take.
        :param seed: Seed for the random number generator.
        :param samples: Number of samples to generate.
        :param safety: DEPRECATED/UNUSED - Cannot be disabled.
        :param classifiers: DEPRECATED/UNUSED - Has no effect on image generation.
        :param guidance_preset: Guidance preset to use. See generation.GuidancePreset for supported values.
        :param guidance_cuts: Number of cuts to use for guidance.
        :param guidance_strength: Strength of the guidance. We recommend values in range [0.0,1.0]. A good default is 0.25
        :param guidance_prompt: Prompt to use for guidance, defaults to `prompt` argument (above) if not specified.
        :param guidance_models: Models to use for guidance.
        :return: Generator of Answer objects.
        """
        if (prompt is None) and (init_image is None):
            raise ValueError("prompt and/or init_image must be provided")

        if (mask_image is not None) and (init_image is None):
            raise ValueError(
                "If mask_image is provided, init_image must also be provided"
            )

        if not seed:
            seed = [random.randrange(0, 4294967295)]
        elif isinstance(seed, int):
            seed = [seed]
        else:
            seed = list(seed)

        prompts: List[generation.Prompt] = []
        if any(isinstance(prompt, t) for t in (str, generation.Prompt)):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, str):
                p = generation.Prompt(text=p)
                if clip_layer:
                    p.parameters.clip_layer = clip_layer
            elif not isinstance(p, generation.Prompt):
                raise TypeError("prompt must be a string or generation.Prompt object")
            prompts.append(p)

        if negative_prompt:
            prompts += [
                generation.Prompt(
                    text=negative_prompt,
                    parameters=generation.PromptParameters(weight=-1),
                )
            ]

        sampler_parameters: dict[str, Any] = dict(cfg_scale=cfg_scale)

        if eta:
            sampler_parameters["eta"] = eta
        if noise_type:
            sampler_parameters["noise_type"] = noise_type

        if churn:
            churn_parameters = dict(churn=churn)

            if churn_tmin:
                churn_parameters["churn_tmin"] = churn_tmin
            if churn_tmax:
                churn_parameters["churn_tmax"] = churn_tmax

            sampler_parameters["churn"] = generation.ChurnSettings(**churn_parameters)

        sigma_parameters = {}

        if sigma_min:
            sigma_parameters["sigma_min"] = sigma_min
        if sigma_max:
            sigma_parameters["sigma_max"] = sigma_max
        if karras_rho:
            sigma_parameters["karras_rho"] = karras_rho

        sampler_parameters["sigma"] = generation.SigmaParameters(**sigma_parameters)

        step_parameters = dict(
            scaled_step=0, sampler=generation.SamplerParameters(**sampler_parameters)
        )

        init_image_prompt = None

        if init_image is not None:
            # NB: Specifying schedule when there's no init image causes washed out results
            step_parameters["schedule"] = generation.ScheduleParameters(
                start=start_schedule,
                end=end_schedule,
            )
            init_image_prompt = image_to_prompt(init_image, init=True)
            prompts += [init_image_prompt]

            if mask_image is not None:
                prompts += [image_to_prompt(mask_image, mask=True)]

            elif mask_from_image_alpha:
                mask_prompt = ref_to_prompt(
                    init_image_prompt.artifact.uuid, type=generation.ARTIFACT_MASK
                )
                mask_prompt.artifact.adjustments.append(
                    generation.ImageAdjustment(
                        channels=generation.ImageAdjustment_Channels(
                            r=generation.CHANNEL_A,
                            g=generation.CHANNEL_A,
                            b=generation.CHANNEL_A,
                            a=generation.CHANNEL_DISCARD,
                        )
                    )
                )
                mask_prompt.artifact.adjustments.append(
                    generation.ImageAdjustment(
                        invert=generation.ImageAdjustment_Invert()
                    )
                )
                mask_prompt.artifact.adjustments.append(
                    generation.ImageAdjustment(
                        blur=generation.ImageAdjustment_Gaussian(
                            sigma=32, direction=generation.DIRECTION_UP
                        )
                    )
                )

                prompts += [mask_prompt]

        if hint_images:
            for hint in hint_images:
                if "image" not in hint:
                    if init_image_prompt is None:
                        raise ValueError(
                            "Can't use hint_from_init without also passing init_image"
                        )

                    hint_prompt = ref_to_prompt(
                        init_image_prompt.artifact.uuid,
                        type=generation.ARTIFACT_HINT_IMAGE,
                    )

                    hint_prompt.echo_back = True
                    hint_prompt.artifact.hint_image_type = hint["hint_type"]
                    hint_prompt.parameters.weight = hint["weight"]
                    hint_prompt.parameters.priority = hint["prioriy"]

                    add_converter_to_hint_image_prompt(
                        hint_prompt, hint["remove_bg"], hint["converter"], hint["args"]
                    )
                else:
                    hint_prompt = hint_image_to_prompt(**hint)

                prompts += [hint_prompt]

        if lora:
            for path, weights in lora:
                prompts += [lora_to_prompt(path, weights, from_cache=from_cache)]

        if ti:
            for path, overrides in ti:
                prompts += ti_to_prompts(path, overrides, from_cache=from_cache)

        if guidance_prompt:
            if isinstance(guidance_prompt, str):
                guidance_prompt = generation.Prompt(text=guidance_prompt)
            elif not isinstance(guidance_prompt, generation.Prompt):
                raise ValueError("guidance_prompt must be a string or Prompt object")
        # if guidance_strength == 0.0:
        #    guidance_strength = None

        # Build our CLIP parameters
        if (
            guidance_preset is not generation.GUIDANCE_PRESET_NONE
            or guidance_strength is not None
        ):
            # to do: make it so user can override this
            # step_parameters['sampler']=None

            if guidance_models:
                guiders = [generation.Model(alias=model) for model in guidance_models]
            else:
                guiders = None

            if guidance_cuts:
                cutouts = generation.CutoutParameters(count=guidance_cuts)
            else:
                cutouts = None

            step_parameters["guidance"] = generation.GuidanceParameters(
                guidance_preset=guidance_preset,
                instances=[
                    generation.GuidanceInstanceParameters(
                        guidance_strength=guidance_strength,
                        models=guiders,
                        cutouts=cutouts,
                        prompt=guidance_prompt,
                    )
                ],
            )

        if hires_fix is None and hires_oos_fraction is not None:
            hires_fix = True

        hires = None

        if hires_fix is not None:
            hires_params: dict[str, bool | float] = dict(enable=hires_fix)
            if hires_oos_fraction is not None:
                hires_params["oos_fraction"] = hires_oos_fraction

            hires = generation.HiresFixParameters(**hires_params)

        tiling_params = {}
        if tiling == "xy" or tiling == "yes":
            tiling_params["tiling"] = True
        elif tiling == "x":
            tiling_params["tiling_x"] = True
        elif tiling == "y":
            tiling_params["tiling_y"] = True

        image_parameters = generation.ImageParameters(
            transform=generation.TransformType(diffusion=sampler),
            seed=seed,
            steps=steps,
            samples=samples,
            parameters=[generation.StepParameter(**step_parameters)],
            hires=hires,
            **tiling_params,
        )

        if height is not None:
            image_parameters.height = height
        if width is not None:
            image_parameters.width = width

        if as_async:
            return self.emit_async_request(
                prompt=prompts,
                image_parameters=image_parameters,
                accept_webp=accept_webp,
            )
        else:
            return self.emit_request(
                prompt=prompts,
                image_parameters=image_parameters,
                accept_webp=accept_webp,
            )

    # The motivation here is to facilitate constructing requests by passing protobuf objects directly.
    def emit_request(
        self,
        prompt: generation.Prompt,
        image_parameters: generation.ImageParameters,
        engine_id: str = None,
        request_id: str = None,
        accept_webp: bool = True,
    ):
        if not request_id:
            request_id = str(uuid.uuid4())
        if not engine_id:
            engine_id = self.engine

        extra_kwargs = {}
        if accept_webp:
            extra_kwargs["accept"] = "image/webp, image/png"

        rq = generation.Request(
            engine_id=engine_id,
            request_id=request_id,
            prompt=prompt,
            image=image_parameters,
            **extra_kwargs,
        )

        # with open("request.json", "w") as f:
        #     json.dump(MessageToDict(rq), f, indent=2)

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        answers = self.stub.Generate(rq, **self.grpc_args)

        def cancel_request(unused_signum, unused_frame):
            print("Cancelling")
            answers.cancel()
            sys.exit(0)

        signal.signal(signal.SIGINT, cancel_request)

        for answer in answers:
            duration = time.time() - start
            if self.verbose:
                if len(answer.artifacts) > 0:
                    artifact_ts = [
                        generation.ArtifactType.Name(artifact.type)
                        for artifact in answer.artifacts
                    ]
                    logger.info(
                        f"Got {answer.answer_id} with {artifact_ts} in "
                        f"{duration:0.2f}s"
                    )
                else:
                    logger.info(
                        f"Got keepalive {answer.answer_id} in " f"{duration:0.2f}s"
                    )

            yield answer
            start = time.time()

    # The motivation here is to facilitate constructing requests by passing protobuf objects directly.
    def emit_async_request(
        self,
        prompt: generation.Prompt,
        image_parameters: generation.ImageParameters,
        engine_id: str = None,
        request_id: str = None,
        accept_webp: bool = True,
    ):
        if not request_id:
            request_id = str(uuid.uuid4())
        if not engine_id:
            engine_id = self.engine

        extra_kwargs = {}
        if accept_webp:
            extra_kwargs["accept"] = "image/webp, image/png"

        rq = generation.Request(
            engine_id=engine_id,
            request_id=request_id,
            prompt=prompt,
            image=image_parameters,
            **extra_kwargs,
        )

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        handle = self.stub.AsyncGenerate(rq, **self.grpc_args)

        print(handle)

        def cancel_request(unused_signum, unused_frame):
            print("Cancelling")
            self.stub.AsyncCancel(handle)
            sys.exit(0)

        signal.signal(signal.SIGINT, cancel_request)

        while True:
            answers = self.stub.AsyncResult(handle)

            for answer in answers.answer:
                yield answer

            if answers.complete:
                if answers.status.code:
                    raise GrpcAsyncError(answers.status.code, answers.status.message)

                print("Done")
                break

            time.sleep(1)


if __name__ == "__main__":
    # Set up logging for output to console.
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    STABILITY_HOST = os.getenv("STABILITY_HOST", "grpc.stability.ai:443")
    STABILITY_KEY = os.getenv("STABILITY_KEY", "")

    if not STABILITY_HOST:
        logger.warning("STABILITY_HOST environment variable needs to be set.")
        sys.exit(1)

    if not STABILITY_KEY:
        logger.warning(
            "STABILITY_KEY environment variable needs to be set. You may"
            " need to login to the Stability website to obtain the"
            " API key."
        )
        sys.exit(1)

    # CLI parsing
    parser = ArgumentParser()
    parser.add_argument(
        "--height", "-H", type=int, default=None, help="[512] height of image"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=None, help="[512] width of image"
    )
    parser.add_argument(
        "--start_schedule",
        type=float,
        default=0.5,
        help="[0.5] start schedule for init image (must be greater than 0, 1 is full strength text prompt, no trace of image)",
    )
    parser.add_argument(
        "--end_schedule",
        type=float,
        default=0.01,
        help="[0.01] end schedule for init image",
    )
    parser.add_argument(
        "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor"
    )
    parser.add_argument(
        "--guidance_strength",
        "-G",
        type=float,
        default=None,
        help="[0.0] CLIP Guidance scale factor. We recommend values in range [0.0,1.0]. A good default is 0.25",
    )
    parser.add_argument(
        "--sampler",
        "-A",
        type=str,
        default="k_lms",
        help="[k_lms] (" + ", ".join(SAMPLERS.keys()) + ")",
    )
    parser.add_argument(
        "--eta",
        "-E",
        type=float,
        default=None,
        help="[None] ETA factor (for DDIM scheduler)",
    )
    parser.add_argument(
        "--churn",
        type=float,
        default=None,
        help="[None] churn factor (for Euler, Heun, DPM2 scheduler)",
    )
    parser.add_argument(
        "--churn_tmin",
        type=float,
        default=None,
        help="[None] churn sigma minimum (for Euler, Heun, DPM2 scheduler)",
    )
    parser.add_argument(
        "--churn_tmax",
        type=float,
        default=None,
        help="[None] churn sigma maximum (for Euler, Heun, DPM2 scheduler)",
    )
    parser.add_argument(
        "--sigma_min", type=float, default=None, help="[None] use this sigma min"
    )
    parser.add_argument(
        "--sigma_max", type=float, default=None, help="[None] use this sigma max"
    )
    parser.add_argument(
        "--karras_rho",
        type=float,
        default=None,
        help="[None] use Karras sigma schedule with this Rho",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="normal",
        help="[normal] (" + ", ".join(NOISE_TYPES.keys()) + ")",
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=50, help="[50] number of steps"
    )
    parser.add_argument("--seed", "-S", type=int, default=0, help="random seed to use")
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default="generation_",
        help="output prefixes for artifacts",
    )
    parser.add_argument(
        "--no-store", action="store_true", help="do not write out artifacts"
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=1, help="number of samples to generate"
    )
    parser.add_argument("--show", action="store_true", help="open artifacts using PIL")
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        help="engine to use for inference",
        default="stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--init_image",
        "-i",
        type=str,
        help="Init image",
    )
    parser.add_argument(
        "--mask_image",
        "-m",
        type=str,
        help="Mask image",
    )
    parser.add_argument(
        "--mask_from_image_alpha",
        "-a",
        action="store_true",
        help="Get the mask from the image alpha channel, rather than a seperate image",
    )
    parser.add_argument(
        "--negative_prompt",
        "-N",
        type=str,
        help="Negative Prompt",
    )
    parser.add_argument(
        "--clip_layer",
        type=int,
        help="Set the clip layer skip (1 is no skip, 2 is skip the first layer, and so on)",
    )
    parser.add_argument(
        "--hires_fix",
        action=BooleanOptionalAction,
        help="Enable or disable the hires fix for images above the 'natural' size of the model",
    )
    parser.add_argument(
        "--hires_oos_fraction",
        type=float,
        help="0..1, how out-of-square the area that's considered when doing a non-square hires fix should be. Low values risk more issues, high values zoom in more.",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        choices=["x", "y", "xy", "no", "yes"],
        help="Select one or both axis to tile on",
    )
    parser.add_argument(
        "--hint_image",
        action="append",
        help="Provide a hint image, in type:path[:weight][:priority] format (priority is balanced, hint or prompt)",
    )
    parser.add_argument(
        "--hint_from_image",
        action="append",
        help="Provide a image to be converted to a hint image, in [nobg:]type[:converter_id]:path[:weight][:priority] format",
    )
    parser.add_argument(
        "--hint_from_init",
        action="append",
        help="Provide a hint image by converting the init_image, in [nobg:]type[:converter_id][:weight][:priority] format",
    )
    parser.add_argument(
        "--lora",
        action="append",
        help="Add a Lora (cloneofsimo, diffusers or kohya-ss format). Either a path, or path:unet_weight or path:unet_weight:text_encode_weight (i.e. ./lora_weight.safetensors:0.5:0.5)",
    )
    parser.add_argument(
        "--ti",
        action="append",
        help="Add a Textual Inversion. Either as a path, or path:token[:token] to override the tokens used (i.e. ./learned_embeds.bin:<token>)",
    )
    parser.add_argument(
        "--list_engines",
        "-L",
        action="store_true",
        help="Print a list of the engines available on the server",
    )
    parser.add_argument(
        "--list_upscalers",
        action="store_true",
        help="Print a list of the upscaler engines available on the server",
    )
    parser.add_argument(
        "--grpc_web",
        action="store_true",
        help="Use GRPC-WEB to connect to the server (instead of GRPC)",
    )
    parser.add_argument(
        "--accept_webp",
        action=BooleanOptionalAction,
        default=True,
        help="Accept webp responses from server (when server supports it)",
    )
    parser.add_argument("--as_async", action="store_true", help="Run asyncronously")
    parser.add_argument("prompt", nargs="*")
    args = parser.parse_args()

    stability_api = StabilityInference(
        STABILITY_HOST,
        STABILITY_KEY,
        proto="grpc-web" if args.grpc_web else "grpc",
        engine=args.engine,
        verbose=True,
    )

    if args.list_engines:
        stability_api.list_engines()
        sys.exit(0)

    if args.list_upscalers:
        stability_api.list_engines(task_group=engines.UPSCALE)
        sys.exit(0)

    if not args.prompt and not args.init_image:
        logger.warning("prompt or init image must be provided")
        parser.print_help()
        sys.exit(1)
    else:
        args.prompt = " ".join(args.prompt)

    if args.init_image:
        args.init_image = Image.open(args.init_image)

    if args.mask_image:
        args.mask_image = Image.open(args.mask_image)

    lora = []
    if args.lora:
        for path in args.lora:
            parts = path.split(":")

            path_parts = []
            while parts and not floatlike(parts[0]):
                path_parts += [parts.pop(0)]
            path = ":".join(path_parts)

            weights = [float(weight) for weight in parts]

            print("Lora", path, weights)
            lora.append((path, weights))

    ti = []
    if args.ti:
        for path in args.ti:
            parts = path.split(":")
            if parts[0] == "https" or parts[0] == "file":
                path, tokens = parts[0] + ":" + parts[1], parts[2:]
            else:
                path, tokens = parts[0], parts[1:]

            ti.append((path, tokens))

    def parse_hint(hint, path, converter):
        args = {}

        if hint.endswith(")"):
            hint, argstr = hint.split("(", 1)
            argstr = argstr[:-1]

            args = yaml.load(
                "{" + argstr.replace("=", ": ") + "}", Loader=yaml.SafeLoader
            )

        parts = hint.split(":")

        remove_bg = False
        if parts[0] == "nobg":
            parts.pop(0)
            remove_bg = True

        priority = generation.HINT_BALANCED
        if parts[-1] in {"balanced", "prompt", "hint"}:
            if parts[-1] == "balanced":
                priority = generation.HINT_BALANCED
            elif parts[-1] == "prompt":
                priority = generation.HINT_PRIORITISE_PROMPT
            elif parts[-1] == "hint":
                priority = generation.HINT_PRIORITISE_HINT
            parts = parts[:-1]

        try:
            weight = float(parts[-1])
            parts = parts[:-1]
        except ValueError:
            weight = 1.0

        hint_info = {
            "hint_type": parts.pop(0),
            "remove_bg": remove_bg,
            "weight": weight,
            "priority": priority,
            "args": args,
        }

        if path:
            if not parts:
                raise ValueError(
                    "No path provided for hint - did you mean hint_from_init?"
                )
            hint_info["image"] = Image.open(parts.pop())

        if converter:
            hint_info["converter"] = parts[0] if parts else True

        return hint_info

    hint_images = []
    if args.hint_image:
        for hint in args.hint_image:
            hint_images.append(parse_hint(hint, path=True, converter=False))

    if args.hint_from_image:
        for hint in args.hint_from_image:
            hint_images.append(parse_hint(hint, path=True, converter=True))

    if args.hint_from_init:
        for hint in args.hint_from_init:
            hint_images.append(parse_hint(hint, path=False, converter=True))

    request = {
        "negative_prompt": args.negative_prompt,
        "clip_layer": args.clip_layer,
        "height": args.height,
        "width": args.width,
        "start_schedule": args.start_schedule,
        "end_schedule": args.end_schedule,
        "cfg_scale": args.cfg_scale,
        "guidance_strength": args.guidance_strength,
        "sampler": get_sampler_from_str(args.sampler),
        "eta": args.eta,
        "churn": args.churn,
        "churn_tmin": args.churn_tmin,
        "churn_tmax": args.churn_tmax,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "karras_rho": args.karras_rho,
        "noise_type": get_noise_type_from_str(args.noise_type),
        "steps": args.steps,
        "seed": args.seed,
        "samples": args.num_samples,
        "init_image": args.init_image,
        "mask_image": args.mask_image,
        "mask_from_image_alpha": args.mask_from_image_alpha,
        "hires_fix": args.hires_fix,
        "hires_oos_fraction": args.hires_oos_fraction,
        "tiling": args.tiling,
        "hint_images": hint_images,
        "lora": lora,
        "ti": ti,
        "as_async": args.as_async,
    }

    try:
        answers = stability_api.generate(
            args.prompt, **request, from_cache=True, accept_webp=args.accept_webp
        )

        artifacts = process_artifacts_from_answers(
            args.prefix, answers, write=not args.no_store, verbose=True
        )

        if args.show:
            for artifact in open_images(artifacts, verbose=True):
                pass
        else:
            for artifact in artifacts:
                pass
    except Exception as e:
        if (
            isinstance(e, grpc.Call | GrpcAsyncError)
            and e.code() is grpc.StatusCode.FAILED_PRECONDITION
        ):
            answers = stability_api.generate(args.prompt, **request, from_cache=False)

            artifacts = process_artifacts_from_answers(
                args.prefix, answers, write=not args.no_store, verbose=True
            )

            if args.show:
                for artifact in open_images(artifacts, verbose=True):
                    pass
            else:
                for artifact in artifacts:
                    pass
        else:
            raise e
