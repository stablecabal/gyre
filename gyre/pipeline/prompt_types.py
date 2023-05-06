from dataclasses import dataclass
from typing import Literal

import torch
from PIL.Image import Image as PILImage

ImageLike = torch.Tensor | PILImage


@dataclass
class PromptFragment:
    prompt: str
    weight: float = 1.0


ClipLayer = int | Literal["final", "penultimate"] | None


def normalise_clip_layer(clip_layer: ClipLayer, default: ClipLayer = None):
    # Handle default
    if clip_layer is None:
        clip_layer = default

    # Absolute if int
    if isinstance(clip_layer, int):
        clip_layer = abs(clip_layer)

    # Now convert to the standard representation
    if clip_layer == 0 or clip_layer == 1 or clip_layer == "final":
        return "final"
    elif clip_layer == 2 or clip_layer == "penultimate":
        return "penultimate"
    else:
        return clip_layer


@dataclass
class Prompt:
    fragments: list[PromptFragment]
    clip_layer: ClipLayer = None

    @classmethod
    def from_string(cls, string, weight=1.0, **kwargs) -> "Prompt":
        return Prompt(
            fragments=[PromptFragment(prompt=string, weight=weight)], **kwargs
        )

    @property
    def weighted(self):
        return any((frag.weight != 1.0 for frag in self.fragments))

    def as_tokens(self):
        return [(frag.prompt, frag.weight) for frag in self.fragments]

    def as_unweighted_string(self):
        return " ".join((frag.prompt for frag in self.fragments))


PromptLike = Prompt | str


class MismatchedClipLayer(ValueError):
    pass


@dataclass
class PromptBatch:
    prompts: list[Prompt]

    @classmethod
    def from_alike(cls, prompt: "PromptBatchLike") -> "PromptBatch":

        if isinstance(prompt, PromptBatch):
            return prompt

        if isinstance(prompt, list):
            prompts = [
                Prompt.from_string(item) if isinstance(item, str) else item
                for item in prompt
            ]

        elif isinstance(prompt, str):
            prompts = [Prompt.from_string(prompt)]

        else:
            prompts = [prompt]

        return PromptBatch(prompts)

    def __len__(self):
        return len(self.prompts)

    @property
    def weighted(self):
        return any((prompt.weighted for prompt in self.prompts))

    def as_tokens(self):
        return [prompt.as_tokens() for prompt in self.prompts]

    def as_unweighted_string(self):
        return [prompt.as_unweighted_string() for prompt in self.prompts]

    def chunk(self):
        return [PromptBatch(prompts=[prompt]) for prompt in self.prompts]

    def clip_layer(self, default: ClipLayer = "final"):
        clip_layers = {
            normalise_clip_layer(prompt.clip_layer, default) for prompt in self.prompts
        }
        if len(clip_layers) == 1:
            return clip_layers.pop()

        raise MismatchedClipLayer("More than one clip layer in batch")


PromptBatchLike = Prompt | str | list[Prompt] | list[str] | PromptBatch


HintPriority = Literal["balanced", "prompt", "hint"]


@dataclass
class HintImage:
    image: ImageLike
    hint_type: str
    weight: float = 1.0
    priority: HintPriority = "balanced"
    clip_layer: ClipLayer = None
