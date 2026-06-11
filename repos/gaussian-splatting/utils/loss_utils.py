#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def weighted_l1_loss(pred, gt, weight_map=None):
    """Per-pixel weighted L1 loss. Falls back to standard L1 when weight_map is None."""
    diff = torch.abs(pred - gt)
    if weight_map is None:
        return diff.mean()
    # weight_map: (H, W) -> broadcast over channels (C, H, W)
    if weight_map.dim() == 2:
        weight_map = weight_map.unsqueeze(0)
    return (diff * weight_map).sum() / (weight_map.sum() * pred.shape[-3] + 1e-8)


def _ssim_map(img1, img2, window_size=11):
    """Compute per-pixel SSIM map (unreduced)."""
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1_val = 0.01 ** 2
    C2_val = 0.03 ** 2

    ssim_map_val = ((2 * mu1_mu2 + C1_val) * (2 * sigma12 + C2_val)) / \
                   ((mu1_sq + mu2_sq + C1_val) * (sigma1_sq + sigma2_sq + C2_val))
    return ssim_map_val  # (1, C, H, W) or (C, H, W)


def weighted_ssim(img1, img2, weight_map=None, window_size=11):
    """Compute SSIM with optional per-pixel weighting.

    Returns a scalar SSIM value (higher = more similar).
    """
    smap = _ssim_map(img1, img2, window_size)  # (C, H, W)
    if weight_map is None:
        return smap.mean()
    if weight_map.dim() == 2:
        weight_map = weight_map.unsqueeze(0)
    return (smap * weight_map).sum() / (weight_map.sum() * smap.shape[-3] + 1e-8)


def huber_loss(pred, gt, delta=0.05):
    """Standard Huber loss for robust pose gradients."""
    diff = pred - gt
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return (0.5 * quadratic.pow(2) + delta * linear).mean()


def pearson_depth_loss(rendered_depth, mono_depth, mask=None):
    """Scale-and-shift invariant depth loss using Pearson correlation.

    Returns 1 - corr(rendered, mono) in [0, 2]. 0 = perfect correlation.
    """
    if mask is not None:
        r = rendered_depth[mask > 0.5]
        m = mono_depth[mask > 0.5]
    else:
        r = rendered_depth.flatten()
        m = mono_depth.flatten()
    if r.numel() < 10:
        return torch.tensor(0.0, device=rendered_depth.device, requires_grad=True)
    x = r - r.mean()
    y = m - m.mean()
    cov = (x * y).mean()
    std_x = torch.sqrt((x * x).mean() + 1e-8)
    std_y = torch.sqrt((y * y).mean() + 1e-8)
    return 1.0 - cov / (std_x * std_y)


def pearson_depth_loss_weighted(rendered_depth, mono_depth, weight, mask=None):
    """Confidence-weighted Pearson correlation loss.

    Each pixel contributes proportional to `weight` (per-pixel confidence in [0,1]).
    Computes weighted Pearson correlation; returns 1 - corr_w in [0, 2].

    weight: same shape as rendered_depth, non-negative.
    mask: binary; if provided, restricts to mask > 0.5 (in addition to weighting).
    """
    if mask is not None:
        r = rendered_depth[mask > 0.5]
        m = mono_depth[mask > 0.5]
        w = weight[mask > 0.5]
    else:
        r = rendered_depth.flatten()
        m = mono_depth.flatten()
        w = weight.flatten()
    if r.numel() < 10:
        return torch.tensor(0.0, device=rendered_depth.device, requires_grad=True)
    # Clamp weights non-negative; small floor to avoid div-by-zero
    w = torch.clamp(w, min=0.0)
    w_sum = w.sum() + 1e-8
    # Weighted means
    mean_r = (w * r).sum() / w_sum
    mean_m = (w * m).sum() / w_sum
    # Weighted centered values
    x = r - mean_r
    y = m - mean_m
    # Weighted cov / var
    cov_w = (w * x * y).sum() / w_sum
    var_x = (w * x * x).sum() / w_sum + 1e-8
    var_y = (w * y * y).sum() / w_sum + 1e-8
    corr_w = cov_w / torch.sqrt(var_x * var_y)
    return 1.0 - corr_w


def wavelet_loss(rendered, gt, lambda_sparse=0.01):
    """Differentiable Haar wavelet frequency regularization.

    Decomposes rendered and GT into LL/LH/HL/HH bands via 2x2 Haar filters.
    Supervises LL (low-freq) with L1, penalizes HF bands for sparsity.
    Uses groups=C to process each channel independently.
    """
    C = rendered.shape[0]  # 3 for RGB
    device = rendered.device
    dtype = rendered.dtype

    # Haar wavelet filters (2x2)
    ll = torch.tensor([[1, 1], [1, 1]], device=device, dtype=dtype) / 4.0
    lh = torch.tensor([[-1, -1], [1, 1]], device=device, dtype=dtype) / 4.0
    hl = torch.tensor([[-1, 1], [-1, 1]], device=device, dtype=dtype) / 4.0
    hh = torch.tensor([[1, -1], [-1, 1]], device=device, dtype=dtype) / 4.0

    rendered_4d = rendered.unsqueeze(0)  # (1, C, H, W)
    gt_4d = gt.unsqueeze(0)

    loss = torch.tensor(0.0, device=device, dtype=dtype)
    for filt, is_low in [(ll, True), (lh, False), (hl, False), (hh, False)]:
        # (C, 1, 2, 2) grouped kernel — each channel gets its own filter
        w = filt.unsqueeze(0).unsqueeze(0).expand(C, 1, 2, 2)
        band_r = F.conv2d(rendered_4d, w, stride=2, groups=C)
        band_g = F.conv2d(gt_4d, w, stride=2, groups=C)
        if is_low:
            loss = loss + torch.abs(band_r - band_g).mean()
        else:
            loss = loss + lambda_sparse * torch.abs(band_r - band_g).mean()
    return loss
