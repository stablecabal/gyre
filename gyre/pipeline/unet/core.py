from typing import Any

import torch

from gyre.pipeline.unet.types import (
    CFGMeta,
    CFGUNet,
    DiffusersUNet,
    EpsTensor,
    ScheduleTimestep,
    XtTensor,
)


class UnetWithExtraChannels:
    def __init__(self, unet: CFGUNet, extra_channels: torch.Tensor):
        self.unet = unet
        self.extra_channels = extra_channels
        self.extra_channels_cfg = torch.cat([extra_channels] * 2)

    def __call__(self, latents: XtTensor, t: ScheduleTimestep, **kwargs) -> EpsTensor:
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
    def __init__(self, unet: CFGUNet, controlnets):
        self.unet = unet
        self.controlnets = controlnets

    def __call__(self, latents: XtTensor, t: ScheduleTimestep, **kwargs) -> EpsTensor:
        cnargs = {
            "encoder_hidden_states": kwargs.get("encoder_hidden_states"),
            "cfg_meta": kwargs.get("cfg_meta"),
        }

        residuals = [
            controlnet(latents, t, **cnargs) for controlnet in self.controlnets
        ]

        resargs = {
            "down_block_additional_residuals": [
                sum(i) for i in zip(*[res.down_block_res_samples for res in residuals])
            ],
            "mid_block_additional_residual": sum(
                [res.mid_block_res_sample for res in residuals]
            ),
        }

        return self.unet(latents, t, **kwargs, **resargs)


class AdapterStateList:
    def __init__(self):
        self.items: list[tuple[list[torch.Tensor], bool]] = []

    def append(self, item: list[torch.Tensor], cfg_only: bool):
        self.items.append((item, cfg_only))

    def _zeros_like(self, item):
        if isinstance(item, torch.Tensor):
            return torch.zeros_like(item)
        else:
            return [self._zeros_like(subitem) for subitem in item]

    @property
    def all(self):
        for item, cfg_only in self.items:
            yield item

    @property
    def cfg_only(self):
        for item, cfg_only in self.items:
            yield item if cfg_only else self._zeros_like(item)

    @property
    def either(self):
        for item, cfg_only in self.items:
            yield item if not cfg_only else self._zeros_like(item)


class UNetWithT2I:
    # Standard states can apply to cfg_only or both
    standard_states: dict[CFGMeta, list[torch.Tensor]] | None = None

    # Style states always just apply to cfg_only, and have special padding rules for unconditional side
    style_states: torch.Tensor | None = None

    def _fuse(self, fuser, coadapters, cfg_meta):
        assert fuser is not None, "fuser is missing"

        # Sum each coadapter seperately (to handle multiple of a specific coadapter type)
        summed_coadapters = {}
        for cotype, states in coadapters.items():
            if isinstance(states, AdapterStateList):
                states = list(getattr(states, cfg_meta))

            if not states:
                continue
            elif isinstance(states[0], list):
                summed_coadapters[cotype] = [sum(i) for i in zip(*states)]
            else:
                summed_coadapters[cotype] = torch.cat(states, dim=1)

        # Then run it through the fuser
        if not summed_coadapters:
            return None, None

        return fuser(summed_coadapters)

    def __init__(self, unet: CFGUNet, t2i_adapters):
        self.unet = unet

        standard_states = AdapterStateList()
        style_states = []
        coadapters = {}
        coadapter_types = set()

        fuser = None

        for adapter in t2i_adapters:
            state, cfg_only = adapter(), adapter.cfg_only

            if cotype := adapter.coadapter_type():
                if isinstance(state, list):
                    coadapter_types.add(cfg_only)
                    coadapters.setdefault(cotype, AdapterStateList()).append(
                        state, cfg_only
                    )
                else:
                    coadapters.setdefault(cotype, []).append(state)

                fuser = adapter.fuser
            else:
                if isinstance(state, list):
                    standard_states.append(state, cfg_only)
                else:
                    style_states.append(state)

        standard_states_g = list(standard_states.all)
        standard_states_u = list(standard_states.either)

        if coadapters:
            # Co-adapters aren't really designed to handle cfg_only, but we do our best

            # 1st, if they're all the same type (or all style-type adapters), it's easy
            if len(coadapter_types) <= 1:
                cfg_only = coadapter_types.pop() if coadapter_types else False

                ca_standards, ca_styles = self._fuse(fuser, coadapters, "all")

                if ca_standards is not None:
                    standard_states_g.append(ca_standards)
                    standard_states_u.append(
                        [torch.zeros_like(t) for t in ca_standards]
                        if cfg_only
                        else ca_standards
                    )

                if ca_styles is not None:
                    style_states.append(ca_styles)

            # Otherwise, run twice, once for each side, and take the styles from the not-cfg-only side
            else:
                for cfg in (True, False):
                    # Run it through the fuser
                    ca_standards, ca_styles = self._fuse(
                        fuser, coadapters, "all" if cfg else "either"
                    )

                    if ca_standards is not None:
                        (standard_states_g if cfg else standard_states_u).append(
                            ca_standards
                        )

                    if not cfg and ca_styles is not None:
                        style_states.append(ca_styles)

        if standard_states_g:
            assert len(standard_states_u) == len(standard_states_g)

            self.standard_states = {
                "u": [sum(i) for i in zip(*standard_states_u)],
                "g": [sum(i) for i in zip(*standard_states_g)],
            }  # type: ignore #

            self.standard_dim0 = self.standard_states["g"][0].shape[0]

            self.standard_states["f"] = [
                torch.cat([u, g], dim=0)
                for u, g in zip(self.standard_states["u"], self.standard_states["g"])
            ]

        if style_states:
            self.style_states = torch.cat(style_states, dim=1)
            self.style_dim0 = self.style_states.shape[0]
            self.style_dim1 = self.style_states.shape[1]

    def __call__(self, latents: XtTensor, t: ScheduleTimestep, **kwargs) -> EpsTensor:
        is_f = kwargs["encoder_hidden_states"].shape[0] == self.standard_dim0 * 2
        cfg_meta: CFGMeta = kwargs.get("cfg_meta", "f" if is_f else "g")

        if self.standard_states is not None:
            kwargs["adapter_states"] = self.standard_states[cfg_meta]

        if self.style_states is not None:
            hidden_states = kwargs.pop("encoder_hidden_states")

            if cfg_meta == "f":
                uncond, cond = hidden_states.chunk(2)
            elif cfg_meta == "u":
                uncond, cond = hidden_states, None
            else:
                uncond, cond = None, hidden_states

            res = []

            if uncond is not None:
                res += [torch.cat([uncond, uncond[:, -self.style_dim1 :, :]], dim=1)]
            if cond is not None:
                res += [torch.cat([cond, self.style_states], dim=1)]

            kwargs["encoder_hidden_states"] = torch.cat(res, dim=0)

        return self.unet(latents, t, **kwargs)


class UNetWithEmbeddings:
    def __init__(
        self,
        unet: CFGUNet,
        text_embeddings: torch.Tensor,
        cfg_meta: CFGMeta,
    ):
        self.unet = unet
        self.text_embeddings = text_embeddings
        self.cfg_meta: CFGMeta = cfg_meta

    def __call__(self, latents: XtTensor, t: ScheduleTimestep) -> EpsTensor:
        return self.unet(
            latents,
            t,
            encoder_hidden_states=self.text_embeddings,
            cfg_meta=self.cfg_meta,
        )


class CFGUNetFromDiffusersUNet:
    def __init__(self, unet: DiffusersUNet):
        self.unet = unet

    def __call__(
        self,
        latents: XtTensor,
        t: ScheduleTimestep,
        *,
        cfg_meta: CFGMeta = None,
        **kwargs,
    ) -> EpsTensor:
        return self.unet(latents, t, **kwargs).sample
