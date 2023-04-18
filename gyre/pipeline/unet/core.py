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

        hidden_states = kwargs["encoder_hidden_states"]

        residuals = [
            controlnet(latents, t, encoder_hidden_states=hidden_states)
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
    standard_states: list[torch.Tensor] | None = None
    style_states: torch.Tensor | None = None

    def __init__(self, unet: DiffusersUNet, t2i_adapters):
        self.unet = unet

        standard_states = []
        style_states = []
        coadapters = {}
        fuser = None

        for adapter in t2i_adapters:
            if cotype := adapter.coadapter_type():
                coadapters.setdefault(cotype, []).append(adapter())
                fuser = adapter.fuser
            else:
                state = adapter()
                if isinstance(state, list):
                    standard_states.append(state)
                else:
                    style_states.append(state)

        if coadapters:
            assert fuser is not None, "fuser is missing"

            # Sum each coadapter seperately (to handle multiple of a specific coadapter type)
            summed_coadapters = {}
            for cotype, states in coadapters.items():
                if isinstance(states[0], list):
                    summed_coadapters[cotype] = [sum(i) for i in zip(*states)]
                else:
                    summed_coadapters[cotype] = torch.cat(states, dim=1)

            # Then run it through the fuser
            coadapter_standard_states, coadapter_style_states = fuser(summed_coadapters)
            if coadapter_standard_states is not None:
                standard_states.append(coadapter_standard_states)
            if coadapter_style_states is not None:
                style_states.append(coadapter_style_states)

        if standard_states:
            self.standard_states = [sum(i) for i in zip(*standard_states)]
            self.standard_dim0 = self.standard_states[0].shape[0]

            self.cfg_standard_states = [
                torch.cat([state] * 2, dim=0) for state in self.standard_states
            ]

        if style_states:
            self.style_states = torch.cat(style_states, dim=1)
            self.style_dim0 = self.style_states.shape[0]
            self.style_dim1 = self.style_states.shape[1]

    def __call__(
        self, latents: XtTensor, t: ScheduleTimestep, **kwargs
    ) -> DiffusersUNetOutput:
        if self.standard_states is not None:
            kwargs["adapter_states"] = (
                self.standard_states
                if self.standard_dim0 == kwargs["encoder_hidden_states"].shape[0]
                else self.cfg_standard_states
            )

        if self.style_states is not None:
            hidden_states = kwargs.pop("encoder_hidden_states")

            # TODO this might be wrong with batch size > 1
            if hidden_states.shape[0] == self.style_dim0:
                hidden_states = torch.cat([hidden_states, self.style_states], dim=1)
            else:
                uncond, cond = hidden_states.chunk(2)

                uncond = torch.cat([uncond, uncond[:, -self.style_dim1 :, :]], dim=1)
                cond = torch.cat([cond, self.style_states], dim=1)

                hidden_states = torch.cat([uncond, cond])

            kwargs["encoder_hidden_states"] = hidden_states

        return self.unet(latents, t, **kwargs)


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
