import glob
import os
from typing import Literal

import huggingface_hub
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from gyre.pipeline.model_loader_base import ModelLoaderBase
from gyre.pipeline.upscalers.models.esrgan_plus import RRDBNet_Plus
from gyre.pipeline.upscalers.models.hat_arch import HAT
from gyre.pipeline.upscalers.models.network_swinir import SwinIR

CONFIG_PATTERN = ["*.yaml"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]

DEFAULT_CONFIGS = {}


# Parameters from https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py#L60
# Confirmed same as https://github.com/XPixelGroup/BasicSR/blob/master/inference/inference_esrgan.py#L25
DEFAULT_CONFIGS["esrgan"] = dict(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4,
)

DEFAULT_CONFIGS["esrgan-plus"] = DEFAULT_CONFIGS["esrgan"]

DEFAULT_CONFIGS["esrgan-6B"] = dict(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=6,
    num_grow_ch=32,
    scale=4,
)

# Parameters from https://github.com/JingyunLiang/SwinIR/blob/main/main_test_swinir.py#L144
DEFAULT_CONFIGS["swinir"] = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="1conv",
)

DEFAULT_CONFIGS["swinir-l"] = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
    embed_dim=240,
    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
    mlp_ratio=2,
    upsampler="nearest+conv",
    resi_connection="3conv",
)

# Parameters from https://github.com/XPixelGroup/HAT/blob/main/options/test/HAT_SRx4_ImageNet-pretrain.yml
DEFAULT_CONFIGS["hat"] = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)

# Parameters from https://github.com/XPixelGroup/HAT/blob/main/options/test/HAT-L_SRx4_ImageNet-pretrain.yml#L49
DEFAULT_CONFIGS["hat-l"] = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)


class UpscalerLoader(ModelLoaderBase):

    # This method adapted from https://github.com/xinntao/ESRGAN/blob/master/transer_RRDB_models.py
    # Distributed under Apache-2.0 license
    @classmethod
    def convert_old_esrgan(
        cls, state_dict: dict[str, torch.Tensor], crt_model: torch.nn.Module
    ):
        crt_net = crt_model.state_dict()

        pretrained_net = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                pretrained_net[k[7:]] = v
            else:
                pretrained_net[k] = v

        dest_to_fill = list(crt_net.keys())
        src_to_take = list(pretrained_net.keys())
        errors = []

        def transfer(dk, sk):
            if sk not in pretrained_net:
                errors.append(f"Missing tensor: {sk} not found")
                return

            crt_net[dk] = pretrained_net[sk]
            dest_to_fill.remove(dk)
            src_to_take.remove(sk)

        # directly copy
        for k, v in crt_net.items():
            if k in pretrained_net and pretrained_net[k].size() == v.size():
                transfer(k, k)

        for sk in crt_net.keys():
            if sk.startswith("body."):
                dk = sk.replace(".rdb", ".RDB")
                dk = dk.replace("body.", "model.1.sub.")

                if "conv1x1" not in dk:
                    dk = dk.replace(".weight", ".0.weight")
                    dk = dk.replace(".bias", ".0.bias")

                transfer(sk, dk)

        for end in ["weight", "bias"]:
            transfer(f"conv_first.{end}", f"model.0.{end}")
            transfer(f"conv_body.{end}", f"model.1.sub.23.{end}")
            transfer(f"conv_up1.{end}", f"model.3.{end}")
            transfer(f"conv_up2.{end}", f"model.6.{end}")
            transfer(f"conv_hr.{end}", f"model.8.{end}")
            transfer(f"conv_last.{end}", f"model.10.{end}")

        if src_to_take:
            errors.append(
                "Unconverted tensors remain after converting" + str(src_to_take)
            )
        if dest_to_fill:
            errors.append("Missing tensors after converting" + str(dest_to_fill))

        if errors:
            raise ValueError("\n".join(errors))

        return crt_net

    @classmethod
    def load(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        network: Literal["esrgan", "esrgan-plus", "swinir", "hat"] = "esrgan",
        converter: Literal["esrgan-old"] | None = None,
        **config,
    ):
        config_name = config.pop("config", network)

        if config_name in DEFAULT_CONFIGS:
            config.update(DEFAULT_CONFIGS[config_name])

        try:
            config_path = cls.get_matching_path(
                path, "config", CONFIG_PATTERN, allow_patterns, ignore_patterns
            )
            config.update(cls.load_config(config_path))

        except FileNotFoundError:
            pass

        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        _, state_dict = cls.load_model(model_path)

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]

        if network == "esrgan":
            model = RRDBNet(**config)
        elif network == "esrgan-plus":
            model = RRDBNet_Plus(**config)
        elif network == "swinir":
            model = SwinIR(**config)
        elif network == "hat":
            model = HAT(**config)
        else:
            raise ValueError(f"Unknown upscaler network kind {network}")

        model._scale = config.get("upscale", config.get("scale", 4))
        model._window_size = config.get("window_size", 32)

        if converter == "esrgan-old":
            state_dict = cls.convert_old_esrgan(state_dict, model)

        model.load_state_dict(state_dict)

        if torch_dtype != "auto":
            model.to(torch_dtype)

        model.eval()
        return model

    @classmethod
    def esrgan(cls, *args, **kwargs):
        return cls.load(*args, network="esrgan", **kwargs)

    @classmethod
    def esrgan_old(cls, *args, **kwargs):
        return cls.load(*args, network="esrgan", converter="esrgan-old", **kwargs)

    @classmethod
    def esrgan_plus(cls, *args, **kwargs):
        return cls.load(*args, network="esrgan-plus", converter="esrgan-old", **kwargs)

    @classmethod
    def realesrgan(cls, *args, **kwargs):
        return cls.load(*args, network="esrgan", **kwargs)

    @classmethod
    def realesrgan_6B(cls, *args, **kwargs):
        return cls.load(*args, network="esrgan", config="esrgan-6B", **kwargs)

    @classmethod
    def swinir(cls, *args, **kwargs):
        return cls.load(*args, network="swinir", **kwargs)

    @classmethod
    def swinir_l(cls, *args, **kwargs):
        return cls.load(*args, network="swinir", config="swinir-l", **kwargs)

    @classmethod
    def hat(cls, *args, **kwargs):
        return cls.load(*args, network="hat", **kwargs)

    @classmethod
    def hat_l(cls, *args, **kwargs):
        return cls.load(*args, network="hat", config="hat-l", **kwargs)
