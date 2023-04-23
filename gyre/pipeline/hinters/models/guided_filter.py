
# From https://github.com/perrying/guided-filter-pytorch
# Distributed under MIT license

# Changes:
# - None

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boxfilter import boxfilter2d

class GuidedFilter2d(nn.Module):
    def __init__(self, radius: int, eps: float):
        super().__init__()
        self.r = radius
        self.eps = eps

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps)
        else:
            raise NotImplementedError

class FastGuidedFilter2d(GuidedFilter2d):
    """Fast guided filter"""
    def __init__(self, radius: int, eps: float, s: int):
        super().__init__(radius, eps)
        self.s = s

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps, self.s)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps, self.s)
        else:
            raise NotImplementedError

def guidedfilter2d_color(guide, src, radius, eps, scale=None):
    """guided filter for a color guide image
    
    Parameters
    -----
    guide: (B, 3, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    assert guide.shape[1] == 3
    if src.ndim == 3:
        src = src[:, None]
    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1) # b x 1 x H x W
    ones = torch.ones_like(guide_r)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N # b x 3 x H x W
    mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1) # b x 1 x H x W

    mean_p = boxfilter2d(src, radius) / N # b x C x H x W

    mean_Ip_r = boxfilter2d(guide_r * src, radius) / N # b x C x H x W
    mean_Ip_g = boxfilter2d(guide_g * src, radius) / N # b x C x H x W
    mean_Ip_b = boxfilter2d(guide_b * src, radius) / N # b x C x H x W

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p # b x C x H x W
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p # b x C x H x W
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p # b x C x H x W

    var_I_rr = boxfilter2d(guide_r * guide_r, radius) / N - mean_I_r * mean_I_r + eps # b x 1 x H x W
    var_I_rg = boxfilter2d(guide_r * guide_g, radius) / N - mean_I_r * mean_I_g # b x 1 x H x W
    var_I_rb = boxfilter2d(guide_r * guide_b, radius) / N - mean_I_r * mean_I_b # b x 1 x H x W
    var_I_gg = boxfilter2d(guide_g * guide_g, radius) / N - mean_I_g * mean_I_g + eps # b x 1 x H x W
    var_I_gb = boxfilter2d(guide_g * guide_b, radius) / N - mean_I_g * mean_I_b # b x 1 x H x W
    var_I_bb = boxfilter2d(guide_b * guide_b, radius) / N - mean_I_b * mean_I_b + eps # b x 1 x H x W

    # determinant
    cov_det = var_I_rr * var_I_gg * var_I_bb \
        + var_I_rg * var_I_gb * var_I_rb \
            + var_I_rb * var_I_rg * var_I_gb \
                - var_I_rb * var_I_gg * var_I_rb \
                    - var_I_rg * var_I_rg * var_I_bb \
                        - var_I_rr * var_I_gb * var_I_gb # b x 1 x H x W

    # inverse
    inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rg = - (var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det # b x 1 x H x W
    inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det # b x 1 x H x W
    inv_var_I_gb = - (var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det # b x 1 x H x W
    inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det # b x 1 x H x W

    inv_sigma = torch.stack([
        torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
        torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
        torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
    ], 1).squeeze(-3) # b x 3 x 3 x H x W

    cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1) # b x 3 x C x H x W

    a = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
    b = mean_p - a[:, 0] * mean_I_r - a[:, 1] * mean_I_g - a[:, 2] * mean_I_b # b x C x H x W

    mean_a = torch.stack([boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = torch.stack([F.interpolate(mean_a[:, i], guide.shape[-2:], mode='bilinear') for i in range(3)], 1)
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = torch.einsum("bichw,bihw->bchw", (mean_a, guide)) + mean_b

    return q

def guidedfilter2d_gray(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image
    
    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide*src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide*guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter2d(a, radius) / N
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = mean_a * guide + mean_b
    return q
