from typing import Any

import torch

from gyre.pipeline.unet.types import (
    DiffusersUNet,
    DiffusersUNetOutput,
    EpsTensor,
    ScheduleTimestep,
    XtTensor,
)


class UnetWithExtraChannels:
    def __init__(self, unet, extra_channels: torch.Tensor):
        self.unet = unet
        self.extra_channels = extra_channels
        self.extra_channels_cfg = torch.cat([extra_channels] * 2)

    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, **kwargs
    ) -> DiffusersUNetOutput:
        if latents.shape[0] == self.extra_channels.shape[0]:
            extra_channels = self.extra_channels
        else:
            extra_channels = self.extra_channels_cfg

        expanded_latents = torch.cat(
            [
                latents,
                # Note: these specifically _do not_ get scaled by the
                # current timestep sigma like the latents above have been
                extra_channels,
            ],
            dim=1,
        )

        return self.unet(expanded_latents, t, **kwargs)


class UNetWithControlnet:
    def __init__(self, unet: DiffusersUNet, controlnets):
        self.unet = unet
        self.controlnets = controlnets

    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, **kwargs
    ) -> DiffusersUNetOutput:
        cnargs: dict[str, Any] = {}

        # If we're using an inpaint unet, latents might have too many channels
        cnlatents = latents[:, 0:4]
        hidden_states = kwargs["encoder_hidden_states"]

        residuals = [
            controlnet(cnlatents, t, encoder_hidden_states=hidden_states)
            for controlnet in self.controlnets
        ]

        cnargs = {
            "down_block_additional_residuals": [
                sum(i) for i in zip(*[res.down_block_res_samples for res in residuals])
            ],
            "mid_block_additional_residual": sum(
                [res.mid_block_res_sample for res in residuals]
            ),
        }

        return self.unet(latents, t, **kwargs, **cnargs)


class UNetWithT2I:
    def __init__(self, unet: DiffusersUNet, t2i_adapters):
        self.unet = unet

        individual_adapter_states = [adapter() for adapter in t2i_adapters]
        adapter_states = [sum(i) for i in zip(*individual_adapter_states)]

        self.adapter_states = adapter_states
        self.adapter_states_dim0 = adapter_states[0].shape[0]

        self.cfg_adapter_states = [
            torch.cat([state] * 2, dim=0) for state in adapter_states
        ]

    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, **kwargs
    ) -> DiffusersUNetOutput:
        adapter_states = (
            self.adapter_states
            if self.adapter_states_dim0 == kwargs["encoder_hidden_states"].shape[0]
            else self.cfg_adapter_states
        )

        return self.unet(latents, t, **kwargs, adapter_states=adapter_states)


class UNetWithT2IStyle:
    def __init__(self, unet: DiffusersUNet, t2istyle_adapters):
        self.unet = unet

        individual_adapter_states = [adapter() for adapter in t2istyle_adapters]
        self.adapter_states = torch.cat(individual_adapter_states, dim=1)

    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, **kwargs
    ) -> DiffusersUNetOutput:

        hidden_states = kwargs.pop("encoder_hidden_states")

        # TODO this might be wrong with batch size > 1
        if hidden_states.shape[0] == self.adapter_states.shape[0]:
            hidden_states = torch.cat([hidden_states, self.adapter_states], dim=1)
        else:
            uncond, cond = hidden_states.chunk(2)

            pad_len = self.adapter_states.size(1)
            uncond = torch.cat([uncond, uncond[:, -pad_len:, :]], dim=1)
            cond = torch.cat([cond, self.adapter_states], dim=1)

            hidden_states = torch.cat([uncond, cond])

        return self.unet(latents, t, encoder_hidden_states=hidden_states, **kwargs)


class UNetWithEmbeddings:
    def __init__(
        self,
        unet: DiffusersUNet,
        text_embeddings: torch.Tensor,
    ):
        self.unet = unet
        self.text_embeddings = text_embeddings

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        kwargs: dict[str, Any] = {"encoder_hidden_states": self.text_embeddings}
        return self.unet(latents, t, **kwargs).sample
