# In part based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py
# Copyright notice from that original file below:

# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from contextlib import contextmanager

import torch
import yaml
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
    create_unet_diffusers_config,
    create_vae_diffusers_config,
)
from safetensors.torch import load_file as torch_safe_load_file
from transformers import CLIPTokenizer

from gyre import torch_safe_unpickler
from gyre.constants import GYRE_BASE_PATH

logger = logging.getLogger(__name__)


EXTRA_CONFIG = {
    "v1-inference": ("epsilon", 512),
    "v2-inference": ("epsilon", 512),
    "v2-inference-v": ("v_prediction", 768),
}


def from_pretrained(cls, model, *args, **kwargs):
    method = getattr(cls, "from_pretrained")
    try:
        return method(model, *args, **kwargs, local_files_only=True)
    except Exception:
        return method(model, *args, **kwargs)


class Config:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        r = self._data[k]
        if isinstance(r, dict):
            return Config(r)
        return r

    def __setitem__(self, k, v):
        self._data[k] = v

    def __dir__(self):
        return self._data.keys()

    def __getattr__(self, k):
        r = self._data[k]
        if isinstance(r, dict):
            return Config(r)
        return r

    def __contains__(self, k):
        return k in self._data

    def __repr__(self):
        return self._data.__repr__()


@contextmanager
def local_only(local_only=True):
    from transformers.utils import hub

    orig = hub._is_offline_mode

    try:
        hub._is_offline_mode = local_only
        yield
    finally:
        hub._is_offline_mode = orig


def load_as_models(
    config,
    checkpoint_path: str | None = None,
    safetensors_path: str | None = None,
    blacklist: set[str] | None = None,
    whitelist: set[str] | None = None,
    image_size: int | None = None,
    prediction_type: str | None = None,
    model_type: str | None = None,
    extract_ema: bool = False,
    scheduler_type: str = "pndm",
    num_in_channels: int | None = None,
    upcast_attention: bool | None = None,
    device: str | None = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.nn.Module]:
    if prediction_type == "v-prediction":
        prediction_type = "v_prediction"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if safetensors_path:
        checkpoint = torch_safe_load_file(safetensors_path, device=device)
    elif checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, pickle_module=torch_safe_unpickler
        )
    else:
        raise ValueError("Must provide one of ckpt_path or safetensors_path")

    # Sometimes models don't have the global_step item
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        logger.debug("global_step key not found in model")
        global_step = None

    # NOTE: this while loop isn't great but this controlnet checkpoint has one additional
    # "state_dict" key https://huggingface.co/thibaud/controlnet-canny-sd21
    while "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not os.path.isfile(config):
        dist_config = os.path.join(GYRE_BASE_PATH, "ldm_config", config + ".yaml")

        if os.path.isfile(dist_config):
            prediction_type, image_size = EXTRA_CONFIG.get(config, (None, None))
            config = dist_config
        else:
            raise RuntimeError(
                f"Config {config} doesn't exist and isn't one of the included"
            )

    with open(config, "r") as f:
        original_config = Config(yaml.load(f, Loader=yaml.SafeLoader))

    if num_in_channels is not None:
        original_config["model"]["params"]["unet_config"]["params"][
            "in_channels"
        ] = num_in_channels

    if (
        "parameterization" in original_config["model"]["params"]
        and original_config["model"]["params"]["parameterization"] == "v"
    ):
        if prediction_type is None:
            prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
            logger.warn(f"Guessed prediction type as {prediction_type}")
        if image_size is None:
            image_size = 512 if global_step == 875000 else 768
            logger.warn(f"Guessed native image size as {image_size}")
    else:
        if prediction_type is None:
            prediction_type = "epsilon"
        if image_size is None:
            image_size = 512

    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end

    models = {}

    def should_load(name):
        if whitelist is not None and name not in whitelist:
            return False
        if blacklist is not None and name in blacklist:
            return False
        return True

    # Convert the text model.
    config_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    if model_type is None:
        model_type = config_type
        logger.debug(f"no `model_type` given, `model_type` inferred as: {model_type}")

    # Check the checkpoint contains the stuff we want, and detect if we need to
    # add key prefixes (regular checkpoint) or not (single model pytorch dump)
    single_model = None

    model_keys = {
        "unet": "model.diffusion_model.",
        "vae": "first_stage_model.",
    }

    if model_type == "FrozenOpenCLIPEmbedder":
        model_keys["text_encoder"] = "cond_stage_model.model.text_projection."
    elif model_type == "FrozenCLIPEmbedder":
        model_keys["text_encoder"] = "cond_stage_model.transformer."

    expected = set()
    missing = set()

    for model, prefix in model_keys.items():
        if should_load(model):
            expected |= {model}

            if not any((1 for key in checkpoint.keys() if key.startswith(prefix))):
                missing |= {model}

    if missing:
        if len(expected) == 1:
            single_model = list(expected).pop()
        else:
            raise RuntimeError(
                f"Checkpoint doesn't contain these models: {missing} - make sure you're passing an appropriate whitelist"
            )

    if single_model and single_model != "vae":
        raise NotImplementedError(
            "Only VAE single model checkpoints currently supported"
        )

    if should_load("scheduler"):
        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
        # make sure scheduler works correctly with DDIM
        scheduler.register_to_config(clip_sample=False)

        if scheduler_type == "pndm":
            config = dict(scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(config)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        elif scheduler_type == "ddim":
            scheduler = scheduler
        else:
            raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

        models["scheduler"] = scheduler

    if should_load("unet"):
        # Convert the UNet2DConditionModel model.
        unet_config = create_unet_diffusers_config(
            original_config, image_size=image_size
        )
        unet_config["upcast_attention"] = upcast_attention
        unet = UNet2DConditionModel(**unet_config)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=checkpoint_path, extract_ema=extract_ema
        )

        unet.load_state_dict(converted_unet_checkpoint)
        models["unet"] = unet

    if should_load("vae"):
        # Convert the VAE model.
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)

        if single_model == "vae":
            checkpoint = {f"first_stage_model.{k}": v for k, v in checkpoint.items()}

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)
        models["vae"] = vae

    if model_type == "FrozenOpenCLIPEmbedder":
        if should_load("text_encoder"):
            text_model = convert_open_clip_checkpoint(checkpoint)
            models["text_encoder"] = text_model
        if should_load("tokenizer"):
            tokenizer = from_pretrained(
                CLIPTokenizer, "stabilityai/stable-diffusion-2", subfolder="tokenizer"
            )
            models["tokenizer"] = tokenizer

    elif model_type == "FrozenCLIPEmbedder":
        if should_load("text_encoder"):
            orig_level = logging.getLogger("transformers.modeling_utils").level

            try:
                logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
                text_model = convert_ldm_clip_checkpoint(checkpoint)
            finally:
                logging.getLogger("transformers.modeling_utils").setLevel(orig_level)

            models["text_encoder"] = text_model
        if should_load("tokenizer"):
            tokenizer = from_pretrained(CLIPTokenizer, "openai/clip-vit-large-patch14")
            models["tokenizer"] = tokenizer

    else:
        raise NotImplementedError(f"Can't convert model of type {model_type}")

    if dtype is not None:
        models = {
            k: v.to(dtype) if isinstance(v, torch.nn.Module) else v
            for k, v in models.items()
        }

    return models
