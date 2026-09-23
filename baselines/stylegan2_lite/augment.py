"""Adaptive Discriminator Augmentation (Karras et al., NeurIPS 2020).

This is what makes the baseline StyleGAN2-*ADA* rather than plain StyleGAN2, and
it is the reason the method is a fair comparison here at all: with only 7243
radiographs the discriminator overfits and the GAN diverges without it.

Two pieces, both from the paper:
  1. A stochastic augmentation pipeline applied to BOTH real and fake images
     before the discriminator sees them, every transform applied with
     probability p. Geometric transforms are composed into a single 3x3 matrix
     and applied once with grid_sample (so the image is resampled once, not once
     per transform); colour transforms are composed into a single 4x4 matrix in
     homogeneous RGB and applied as one matmul.
  2. The adaptive controller: r_t = E[sign(D(real))] estimates discriminator
     overfitting, and p is nudged up or down to hold r_t at a target (0.6).

Augmentations implemented: x-flip, 90-degree rotation, integer translation,
isotropic scale, arbitrary rotation, anisotropic scale, fractional translation,
brightness, contrast, luma flip, hue, saturation.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


class AugmentPipe(torch.nn.Module):
    def __init__(self, p=0.0, xflip=1, rotate90=1, xint=1, xint_max=0.125,
                 scale=1, rotate=1, aniso=1, xfrac=1,
                 scale_std=0.2, rotate_max=1.0, aniso_std=0.2, xfrac_std=0.125,
                 brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
                 brightness_std=0.2, contrast_std=0.5, hue_max=1.0, saturation_std=1.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(p)))
        for k, v in dict(xflip=xflip, rotate90=rotate90, xint=xint, scale=scale,
                         rotate=rotate, aniso=aniso, xfrac=xfrac,
                         brightness=brightness, contrast=contrast, lumaflip=lumaflip,
                         hue=hue, saturation=saturation).items():
            setattr(self, k, float(v))
        self.xint_max, self.scale_std, self.rotate_max = xint_max, scale_std, rotate_max
        self.aniso_std, self.xfrac_std = aniso_std, xfrac_std
        self.brightness_std, self.contrast_std = brightness_std, contrast_std
        self.hue_max, self.saturation_std = hue_max, saturation_std

    def _maybe(self, cond_prob, value, default, shape, device):
        """Apply `value` with probability p*cond_prob, else `default`."""
        keep = (torch.rand(shape, device=device) < self.p * cond_prob).float()
        return value * keep + default * (1 - keep)

    def forward(self, images):
        N, C, H, W = images.shape
        dev = images.device
        if float(self.p) == 0:
            return images

        # ---- geometric: compose one 3x3 matrix, resample once -----------------
        G = torch.eye(3, device=dev).unsqueeze(0).repeat(N, 1, 1)

        def scale2d(sx, sy):
            m = torch.eye(3, device=dev).unsqueeze(0).repeat(N, 1, 1)
            m[:, 0, 0] = sx; m[:, 1, 1] = sy; return m

        def translate2d(tx, ty):
            m = torch.eye(3, device=dev).unsqueeze(0).repeat(N, 1, 1)
            m[:, 0, 2] = tx; m[:, 1, 2] = ty; return m

        def rotate2d(theta):
            m = torch.eye(3, device=dev).unsqueeze(0).repeat(N, 1, 1)
            c, s = torch.cos(theta), torch.sin(theta)
            m[:, 0, 0] = c; m[:, 0, 1] = -s; m[:, 1, 0] = s; m[:, 1, 1] = c; return m

        if self.xflip:
            i = torch.floor(torch.rand(N, device=dev) * 2)
            i = self._maybe(self.xflip, i, torch.zeros_like(i), (N,), dev)
            G = G @ scale2d(1 - 2 * i, torch.ones(N, device=dev))
        if self.rotate90:
            i = torch.floor(torch.rand(N, device=dev) * 4)
            i = self._maybe(self.rotate90, i, torch.zeros_like(i), (N,), dev)
            G = G @ rotate2d(-math.pi / 2 * i)
        if self.xint:
            t = (torch.rand(N, 2, device=dev) * 2 - 1) * self.xint_max
            t = self._maybe(self.xint, t, torch.zeros_like(t), (N, 1), dev)
            G = G @ translate2d(t[:, 0], t[:, 1])
        if self.scale:
            s = torch.exp2(torch.randn(N, device=dev) * self.scale_std)
            s = self._maybe(self.scale, s, torch.ones_like(s), (N,), dev)
            G = G @ scale2d(1 / s, 1 / s)
        if self.rotate:
            th = (torch.rand(N, device=dev) * 2 - 1) * (math.pi * self.rotate_max)
            th = self._maybe(self.rotate, th, torch.zeros_like(th), (N,), dev)
            G = G @ rotate2d(-th)
        if self.aniso:
            a = torch.exp2(torch.randn(N, device=dev) * self.aniso_std)
            a = self._maybe(self.aniso, a, torch.ones_like(a), (N,), dev)
            G = G @ scale2d(1 / a, a)
        if self.xfrac:
            t = torch.randn(N, 2, device=dev) * self.xfrac_std
            t = self._maybe(self.xfrac, t, torch.zeros_like(t), (N, 1), dev)
            G = G @ translate2d(t[:, 0], t[:, 1])

        if not torch.allclose(G, torch.eye(3, device=dev).unsqueeze(0).expand_as(G)):
            grid = F.affine_grid(G[:, :2], images.shape, align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear",
                                   padding_mode="reflection", align_corners=False)

        # ---- colour: compose one 4x4 homogeneous matrix ----------------------
        Cm = torch.eye(4, device=dev).unsqueeze(0).repeat(N, 1, 1)
        v = torch.tensor([1, 1, 1, 0], device=dev, dtype=torch.float32) / math.sqrt(3)

        if self.brightness:
            b = torch.randn(N, device=dev) * self.brightness_std
            b = self._maybe(self.brightness, b, torch.zeros_like(b), (N,), dev)
            m = torch.eye(4, device=dev).unsqueeze(0).repeat(N, 1, 1)
            m[:, 0, 3] = m[:, 1, 3] = m[:, 2, 3] = b
            Cm = m @ Cm
        if self.contrast:
            c = torch.exp2(torch.randn(N, device=dev) * self.contrast_std)
            c = self._maybe(self.contrast, c, torch.ones_like(c), (N,), dev)
            m = torch.eye(4, device=dev).unsqueeze(0).repeat(N, 1, 1)
            m[:, 0, 0] = m[:, 1, 1] = m[:, 2, 2] = c
            Cm = m @ Cm
        if self.lumaflip:
            i = torch.floor(torch.rand(N, 1, 1, device=dev) * 2)
            i = i * (torch.rand(N, 1, 1, device=dev) < self.p * self.lumaflip).float()
            Cm = (torch.eye(4, device=dev) - 2 * torch.outer(v, v) * i) @ Cm
        if self.hue and C > 1:
            th = (torch.rand(N, device=dev) * 2 - 1) * (math.pi * self.hue_max)
            th = self._maybe(self.hue, th, torch.zeros_like(th), (N,), dev)
            Cm = _rotate3d(v[:3], th) @ Cm
        if self.saturation:
            s = torch.exp2(torch.randn(N, 1, 1, device=dev) * self.saturation_std)
            keep = (torch.rand(N, 1, 1, device=dev) < self.p * self.saturation).float()
            s = s * keep + (1 - keep)
            Cm = (torch.outer(v, v) + (torch.eye(4, device=dev) - torch.outer(v, v)) * s) @ Cm

        images = images.reshape(N, C, H * W)
        if C == 3:
            images = Cm[:, :3, :3] @ images + Cm[:, :3, 3:]
        else:
            images = images * Cm[:, :1, :1] + Cm[:, :1, 3:]
        return images.reshape(N, C, H, W)


def _rotate3d(axis, angle):
    """4x4 rotation about `axis` by `angle`, batched over angle."""
    N = angle.shape[0]
    x, y, z = axis
    c, s = torch.cos(angle), torch.sin(angle)
    o = 1 - c
    m = torch.eye(4, device=angle.device).unsqueeze(0).repeat(N, 1, 1)
    m[:, 0, 0] = x * x * o + c;     m[:, 0, 1] = x * y * o - z * s; m[:, 0, 2] = x * z * o + y * s
    m[:, 1, 0] = y * x * o + z * s; m[:, 1, 1] = y * y * o + c;     m[:, 1, 2] = y * z * o - x * s
    m[:, 2, 0] = z * x * o - y * s; m[:, 2, 1] = z * y * o + x * s; m[:, 2, 2] = z * z * o + c
    return m


class ADAController:
    """Nudge p to hold r_t = E[sign(D(real))] at `target` (paper Sec. 3)."""
    def __init__(self, pipe, target=0.6, interval=4, kimg=500, batch=32):
        self.pipe, self.target, self.interval = pipe, target, interval
        self.step = batch * interval / (kimg * 1000)
        self.buf = []

    def update(self, d_real_logits):
        self.buf.append(torch.sign(d_real_logits.detach()).mean())
        if len(self.buf) < self.interval:
            return float(self.pipe.p)
        rt = torch.stack(self.buf).mean(); self.buf = []
        adjust = self.step * (1 if rt > self.target else -1)
        self.pipe.p.copy_((self.pipe.p + adjust).clamp(0.0, 1.0))
        return float(self.pipe.p)
