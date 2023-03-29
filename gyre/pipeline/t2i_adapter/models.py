# Original from https://github.com/HimariO/diffusers-t2i-adapter/blob/dcfcce9095d587dddbea12d4612955ff4da662b7/src/diffusers/models/adapter.py
# modify from https://github.com/TencentARC/T2I-Adapter/blob/main/ldm/modules/encoders/adapter.py

import glob
import os
from copy import deepcopy

import huggingface_hub
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .adapter import Adapter, Adapter_light, CoAdapterFuser, LayerNorm, StyleAdapter


class T2iAdapter:
    @classmethod
    def from_state_dict(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        coadapter=False,
        **config,
    ):
        t2i_type = config.pop("type", "main")

        if t2i_type == "main":
            adapter_cls = T2iAdapter_main
        elif t2i_type == "style":
            adapter_cls = T2iAdapter_style
        elif t2i_type == "light":
            adapter_cls = T2iAdapter_light
        elif t2i_type == "fuser":
            adapter_cls = T2iAdapter_fuser
        else:
            raise ValueError(f"Unknown T2i Adapter type {t2i_type}")

        paths = glob.glob("*.pt", root_dir=path) + glob.glob("*.pth", root_dir=path)
        paths = list(
            huggingface_hub.utils.filter_repo_objects(
                paths,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

        if not paths:
            raise RuntimeError(f"No model found for T2iAdapter at {path}")

        adapter = adapter_cls(**{**adapter_cls.default_config, **config})
        adapter.load_state_dict(torch.load(os.path.join(path, paths[0])))

        if coadapter:
            adapter._coadapter_type = coadapter

        if torch_dtype != "auto":

            layernorms = []

            # Kind of a hack - LayerNorm modules need to be left in float32,
            # so make a copy of all the parameters before running .to
            for module in adapter.modules():
                if isinstance(module, LayerNorm):
                    weight, bias = deepcopy(module.weight), deepcopy(module.bias)
                    layernorms.append((module, weight, bias))

            adapter.to(torch_dtype)

            # And now restore those parameters
            for module, weight, bias in layernorms:
                module.weight, module.bias = weight, bias

        adapter.eval()
        return adapter


class T2iAdapter_main(Adapter, T2iAdapter, ModelMixin, ConfigMixin):
    # Defaults from https://github.com/TencentARC/T2I-Adapter/blob/main/ldm/inference_base.py#L228
    default_config = dict(
        cin=int(3 * 64),
        channels=[320, 640, 1280, 1280][:4],
        nums_rb=2,
        ksize=1,
        sk=True,
        use_conv=False,
    )

    @register_to_config
    def __init__(
        self,
        channels=[320, 640, 1280, 1280],
        nums_rb=3,
        cin=64,
        ksize=3,
        sk=False,
        use_conv=True,
        autoinvert=False,
    ):
        self.autoinvert = autoinvert

        super().__init__(
            channels=channels,
            nums_rb=nums_rb,
            cin=cin,
            ksize=ksize,
            sk=sk,
            use_conv=use_conv,
        )

    def forward(self, x):
        if self.autoinvert:
            # If sample is more than 2/3 white, assume it needs inverting
            if x.mean() > 0.66:
                x = 1 - x

        return super().forward(x)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        super().from_state_dict(
            path,
            torch_dtype,
            low_cpu_mem_usage,
            allow_patterns,
            ignore_patterns,
            type="main",
            **config,
        )


class T2iAdapter_style(StyleAdapter, T2iAdapter, ModelMixin, ConfigMixin):
    default_config = dict(
        width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8
    )

    @register_to_config
    def __init__(self, width=1024, context_dim=768, num_head=8, n_layes=3, num_token=4):
        super().__init__(
            width=width,
            context_dim=context_dim,
            num_head=num_head,
            n_layes=n_layes,
            num_token=num_token,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        super().from_state_dict(
            path,
            torch_dtype,
            low_cpu_mem_usage,
            allow_patterns,
            ignore_patterns,
            type="style",
            **config,
        )


class T2iAdapter_light(Adapter_light, T2iAdapter, ModelMixin, ConfigMixin):
    default_config = dict(
        cin=int(3 * 64),
        channels=[320, 640, 1280, 1280][:4],
        nums_rb=4,
    )

    @register_to_config
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64):
        super().__init__(channels=channels, nums_rb=nums_rb, cin=cin)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        super().from_state_dict(
            path,
            torch_dtype,
            low_cpu_mem_usage,
            allow_patterns,
            ignore_patterns,
            type="light",
            **config,
        )


class T2iAdapter_fuser(CoAdapterFuser, T2iAdapter, ModelMixin, ConfigMixin):
    # Defaults from https://github.com/TencentARC/T2I-Adapter/blob/main/app_coadapter.py#L58
    default_config = dict(
        unet_channels=[320, 640, 1280, 1280],
        width=768,
        num_head=8,
        n_layes=3,
    )

    @register_to_config
    def __init__(
        self, unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3
    ):
        super().__init__(
            unet_channels=unet_channels, width=width, num_head=num_head, n_layes=n_layes
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_state_dict(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        allow_patterns=[],
        ignore_patterns=[],
        **config,
    ):
        super().from_state_dict(
            path,
            torch_dtype,
            low_cpu_mem_usage,
            allow_patterns,
            ignore_patterns,
            type="fuser",
            **config,
        )
