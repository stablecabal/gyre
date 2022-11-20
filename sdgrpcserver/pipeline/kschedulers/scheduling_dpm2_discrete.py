# Copyright 2022 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
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

from typing import Optional, Tuple, Union

import numpy as np
import torch

from scipy import integrate

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from .scheduling_utils import KSchedulerMixin


class DPM2DiscreteScheduler(KSchedulerMixin, ConfigMixin):
    """
    A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
    for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L119

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085, #sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.derivatives = []

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=float)
        self.timesteps = torch.from_numpy(timesteps)

        low_idx = np.floor(timesteps).astype(int)
        high_idx = np.ceil(timesteps).astype(int)
        frac = np.mod(timesteps, 1.0)
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = torch.from_numpy(sigmas)

        self.init_noise_sigma = self.sigmas[0]
        self.derivatives = []

    def step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: float,
        sample: Union[torch.FloatTensor, np.ndarray],
        s_churn: float = 0.,
        s_tmin:  float = 0.,
        s_tmax: float = float('inf'),
        s_noise:  float = 1.,
        generator = None,
        noise_predictor = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if not noise_predictor: print("Noise predictor not provided, result will not be correct.")

        index = self.t_to_index(timestep)

        sigma = self.sigmas[index]
        gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.
        eps = torch.randn(sample.size(), dtype=sample.dtype, layout=sample.layout, device=generator.device, generator=generator).to(sample.device) * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        pred_original_sample = sample - sigma_hat * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat
        self.derivatives.append(derivative)

        if self.sigmas[index + 1] == 0:
            dt = self.sigmas[index + 1] - sigma_hat
            sample = sample + derivative * dt
        else:
            sigma_mid = sigma_hat.log().lerp(self.sigmas[index + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = self.sigmas[index + 1] - sigma_hat
            sample_2 = sample + derivative * dt_1

            if noise_predictor:
                model_output_2 = noise_predictor(sample_2, self.sigma_to_t(sigma_mid))
                pred_original_sample_2 = sample_2 - sigma_mid * model_output_2
            else:
                pred_original_sample_2 = sample_2 - sigma_mid * model_output
            
            derivative_2 = (sample_2 - pred_original_sample_2) / sigma_mid
            sample = sample + derivative_2 * dt_2

        prev_sample = sample

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)
