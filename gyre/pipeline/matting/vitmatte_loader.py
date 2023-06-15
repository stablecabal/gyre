import functools
import inspect
from types import SimpleNamespace as SN
from functools import partial
import torch.nn as nn
from gyre.pipeline.model_loader_base import ModelLoaderBase
from gyre.pipeline.matting.matte_anything import (
    ViTMatte,
    MattingCriterion,
    Detail_Capture,
    ViT,
)

VITMATTE_CONFIGS = {
    "S": {},
    "B": {
        "backbone.embed_dim": 768,
        "backbone.num_heads": 12,
        "decoder.in_chans": 768,
    },
}


CONFIG_PATTERN = ["*.yaml"]
MODELS_PATTERN = ["*.safetensors", "*.pt", "*.pth"]


class VitmatteLoader(ModelLoaderBase):
    @classmethod
    def build_vitmatte(cls, **config):
        # Default backbone args from "configs/common/model.py"
        backbone_config = dict(
            in_chans=4,
            img_size=512,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            drop_path_rate=0,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[
                # 2, 5, 8, 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[2, 5, 8, 11],
            use_rel_pos=True,
            out_feature="last_feat",
        )

        # Default decoder args
        decoder_config = dict()

        # Apply any overrides
        for key, value in config.items():
            module_name, key = key.split(".")
            if module_name == "backbone":
                backbone_config[key] = value
            elif module_name == "decoder":
                decoder_config[key] = value
            else:
                raise KeyError(f"Unknown module {module_name}")

        # Build modules
        backbone = ViT(**backbone_config)
        decoder = Detail_Capture(**decoder_config)

        # Built model
        return ViTMatte(
            backbone=backbone,
            criterion=MattingCriterion(
                losses=[
                    "unknown_l1_loss",
                    "known_l1_loss",
                    "loss_pha_laplacian",
                    "loss_gradient_penalty",
                ]
            ),
            pixel_mean=[123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0],
            pixel_std=[58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0],
            input_format="RGB",
            size_divisibility=32,
            decoder=decoder,
        )

    @classmethod
    def load(
        cls,
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        model_base="B",
        **config_overrides,
    ):
        config = VITMATTE_CONFIGS[model_base]
        config.update(config_overrides)

        model = cls.build_vitmatte(**config)

        # Load the actual state
        model_path = cls.get_matching_path(
            path, "model", MODELS_PATTERN, allow_patterns, ignore_patterns
        )
        _, state_dict = cls.load_model(model_path)

        model.load_state_dict(state_dict, strict=True)

        model.eval()
        if torch_dtype != "auto":
            model.to(torch_dtype)

        return model
