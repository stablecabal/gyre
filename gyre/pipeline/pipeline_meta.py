import generation_pb2
import torch

from gyre import images
from gyre.pipeline.prompt_types import Prompt, PromptBatch


def image_to_centered(i):
    if isinstance(i, torch.Tensor):
        i = images.normalise_tensor(i, 3)
        return i * 2 - 1
    return i


def prompt_to_string(p):
    if isinstance(p, Prompt | PromptBatch):
        return p.as_unweighted_string()
    return p


def apply_fields(meta, args):
    fields = meta.get("fields")

    if fields is None:
        return args

    res = {**args}

    for k, c in fields.items():
        if k in res:
            res[k] = c(res[k])

    return res


PIPELINE_META = {}

PIPELINE_META["StableDiffusionUpscalePipeline"] = dict(
    fields=dict(
        image=image_to_centered,
        prompt=prompt_to_string,
        negative_prompt=prompt_to_string,
        eta=lambda eta: 0.0 if eta is None else eta,
    ),
    xformers="manual",
    wrapper="gyre.pipeline.upscalers.diffusers_upscaler_wrapper.DiffusionUpscalerPipelineWrapper",
)

PIPELINE_META["StableDiffusionLatentUpscalePipeline"] = dict(
    fields=dict(
        image=image_to_centered,
        prompt=prompt_to_string,
        negative_prompt=prompt_to_string,
    ),
    xformers="manual",
    diffusers_capable=False,
    kdiffusion_capable=False
    # TODO: This needs to be Diffuser's built in euler implementation
    # schedulers={generation_pb2.SAMPLER_K_EULER},
)


def get_meta(class_or_instance):
    if type(class_or_instance) is not type:
        class_or_instance = type(class_or_instance)

    meta = {}
    name = class_or_instance.__name__

    # Update with external meta
    meta.update(PIPELINE_META.get(name, {}))

    # Update with internal meta
    meta.update(getattr(class_or_instance, "_meta", {}))

    return meta
