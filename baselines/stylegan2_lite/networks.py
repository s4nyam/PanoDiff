"""StyleGAN2 generator/discriminator in plain PyTorch (Karras et al., CVPR 2020).

NVIDIA's release relies on hand-written CUDA kernels (bias_act, upfirdn2d) that
have no working ROCm path on LUMI: with cuDNN benchmarking on, MIOpen autotunes
the 1024^2 convolutions indefinitely; with it off, training runs ~96x too slow.
The *method* has no such constraint, so the architecture is rebuilt here from the
paper using only standard ops.

Faithful to StyleGAN2 in the parts that define it:
  * 8-layer mapping network with pixel-norm on z and w-averaging for truncation
  * weight modulation + demodulation (Sec. 2.2) -- the change that removed the
    droplet artefacts of StyleGAN1's AdaIN
  * per-layer noise injection with a learned scale
  * skip-connection generator / residual discriminator (Sec. 3, config E-F)
  * minibatch-stddev layer in the discriminator epilogue
  * equalised learning rate on every weight (He-init at runtime, not init time)
  * blur (binomial low-pass) on every up/downsample

Substituted: `upfirdn2d` fused resample -> F.interpolate/avg_pool plus an explicit
binomial blur, which is the same operation expressed in stock ops.

Unlike the reference implementation this supports NON-SQUARE output, so the
radiographs train at their native 512x1024 instead of being squashed to a square.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _blur_kernel(device, dtype):
    f = torch.tensor([1., 3., 3., 1.], device=device, dtype=dtype)
    f = f[:, None] * f[None, :]
    return (f / f.sum())[None, None]


def blur2d(x):
    k = _blur_kernel(x.device, x.dtype).expand(x.shape[1], 1, 4, 4)
    return F.conv2d(F.pad(x, (1, 2, 1, 2), mode="reflect"), k, groups=x.shape[1])


class EqLinear(nn.Module):
    """Linear layer with equalised learning rate."""
    def __init__(self, i, o, bias=True, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.scale = lr_mul / math.sqrt(i)
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        b = self.bias * self.lr_mul if self.bias is not None else None
        x = F.linear(x, self.weight * self.scale, b)
        if self.activation == "lrelu":
            x = F.leaky_relu(x, 0.2) * math.sqrt(2)
        return x


class ModulatedConv2d(nn.Module):
    """The core of StyleGAN2: scale the conv weights by the style, then
    demodulate so the output keeps unit variance (paper Eq. 1-3)."""
    def __init__(self, in_ch, out_ch, k, w_dim, demodulate=True, up=False):
        super().__init__()
        self.in_ch, self.out_ch, self.k, self.up = in_ch, out_ch, k, up
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.scale = 1 / math.sqrt(in_ch * k * k)
        self.affine = EqLinear(w_dim, in_ch, bias=True)
        self.affine.bias.data.fill_(1.0)          # styles start at 1
        self.padding = k // 2

    def forward(self, x, w):
        N, C, H, W = x.shape
        s = self.affine(w)                                     # [N, in_ch]
        weight = self.weight[None] * self.scale * s[:, None, :, None, None]
        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * d[:, :, None, None, None]
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            H, W = H * 2, W * 2
        x = x.reshape(1, N * C, H, W)
        weight = weight.reshape(N * self.out_ch, C, self.k, self.k)
        x = F.conv2d(x, weight, padding=self.padding, groups=N)
        x = x.reshape(N, self.out_ch, H, W)
        if self.up:
            x = blur2d(x)
        return x


class SynthesisLayer(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, up=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, w_dim, up=up)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, w, noise=None):
        x = self.conv(x, w)
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3],
                                device=x.device, dtype=x.dtype)
        x = x + self.noise_strength * noise
        return F.leaky_relu(x + self.bias[None, :, None, None], 0.2) * math.sqrt(2)


class ToRGB(nn.Module):
    def __init__(self, in_ch, w_dim, out_ch=3):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, w, skip=None):
        x = self.conv(x, w) + self.bias[None, :, None, None]
        if skip is not None:
            skip = blur2d(F.interpolate(skip, scale_factor=2, mode="nearest"))
            x = x + skip
        return x


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, num_ws=None):
        super().__init__()
        self.z_dim, self.w_dim, self.num_ws = z_dim, w_dim, num_ws
        layers = []
        for i in range(num_layers):
            layers.append(EqLinear(z_dim if i == 0 else w_dim, w_dim,
                                   lr_mul=0.01, activation="lrelu"))
        self.net = nn.Sequential(*layers)
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(self, z, truncation_psi=1.0, update_w_avg=True):
        x = z * torch.rsqrt(z.pow(2).mean(1, keepdim=True) + 1e-8)   # pixel norm
        w = self.net(x)
        if self.training and update_w_avg:
            with torch.no_grad():
                self.w_avg.copy_(w.detach().mean(0).lerp(self.w_avg, 0.995))
        if truncation_psi != 1.0:
            w = self.w_avg.lerp(w, truncation_psi)
        return w.unsqueeze(1).repeat(1, self.num_ws, 1)


class Generator(nn.Module):
    """Skip-connection generator. `base` is the constant input resolution and is
    non-square for 2:1 radiographs (4x8 -> ... -> 512x1024)."""
    def __init__(self, z_dim=512, w_dim=512, img_resolution=(512, 1024),
                 img_channels=3, channel_base=32768, channel_max=512, base=(4, 8)):
        super().__init__()
        self.z_dim, self.w_dim = z_dim, w_dim
        self.img_resolution = img_resolution
        self.base = base
        self.num_blocks = int(math.log2(img_resolution[0] // base[0]))
        def nf(stage):
            return min(channel_max, channel_base // (2 ** (stage + 2)))
        self.const = nn.Parameter(torch.randn(1, nf(0), base[0], base[1]))
        self.conv0 = SynthesisLayer(nf(0), nf(0), w_dim)
        self.torgb0 = ToRGB(nf(0), w_dim, img_channels)
        self.blocks, self.torgbs = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_blocks):
            cin, cout = nf(i), nf(i + 1)
            self.blocks.append(nn.ModuleList([
                SynthesisLayer(cin, cout, w_dim, up=True),
                SynthesisLayer(cout, cout, w_dim)]))
            self.torgbs.append(ToRGB(cout, w_dim, img_channels))
        self.num_ws = 2 + self.num_blocks * 2
        self.mapping = MappingNetwork(z_dim, w_dim, num_ws=self.num_ws)

    def forward(self, z, truncation_psi=1.0, ws=None):
        if ws is None:
            ws = self.mapping(z, truncation_psi)
        x = self.const.repeat(ws.shape[0], 1, 1, 1)
        x = self.conv0(x, ws[:, 0])
        img = self.torgb0(x, ws[:, 1])
        i = 1
        for (c_up, c_same), trgb in zip(self.blocks, self.torgbs):
            x = c_up(x, ws[:, i]); i += 1
            x = c_same(x, ws[:, i]); i += 1
            img = trgb(x, ws[:, i], skip=img)
        return img


class EqConv2d(nn.Module):
    def __init__(self, i, o, k, stride=1, padding=0, activation=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, k, k))
        self.bias = nn.Parameter(torch.zeros(o))
        self.scale = 1 / math.sqrt(i * k * k)
        self.stride, self.padding, self.activation = stride, padding, activation

    def forward(self, x):
        x = F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)
        return F.leaky_relu(x, 0.2) * math.sqrt(2) if self.activation else x


class DBlock(nn.Module):
    """Residual discriminator block (StyleGAN2 config E)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = EqConv2d(in_ch, in_ch, 3, padding=1)
        self.conv1 = EqConv2d(in_ch, out_ch, 3, padding=1)
        self.skip = EqConv2d(in_ch, out_ch, 1, activation=False)

    def forward(self, x):
        y = self.skip(F.avg_pool2d(blur2d(x), 2))
        x = self.conv1(self.conv0(x))
        x = F.avg_pool2d(blur2d(x), 2)
        return (x + y) * (1 / math.sqrt(2))


class Discriminator(nn.Module):
    def __init__(self, img_resolution=(512, 1024), img_channels=3,
                 channel_base=32768, channel_max=512, base=(4, 8), mbstd_group=4):
        super().__init__()
        self.num_blocks = int(math.log2(img_resolution[0] // base[0]))
        def nf(stage):
            return min(channel_max, channel_base // (2 ** (stage + 2)))
        self.from_rgb = EqConv2d(img_channels, nf(self.num_blocks), 1)
        blocks = []
        for i in range(self.num_blocks, 0, -1):
            blocks.append(DBlock(nf(i), nf(i - 1)))
        self.blocks = nn.Sequential(*blocks)
        self.mbstd_group = mbstd_group
        self.conv_out = EqConv2d(nf(0) + 1, nf(0), 3, padding=1)
        self.fc = EqLinear(nf(0) * base[0] * base[1], nf(0), activation="lrelu")
        self.out = EqLinear(nf(0), 1)

    def forward(self, img):
        x = self.blocks(self.from_rgb(img))
        N, C, H, W = x.shape
        G = min(self.mbstd_group, N)
        if N % G == 0:
            y = x.reshape(G, -1, C, H, W)
            y = (y.var(0, unbiased=False) + 1e-8).sqrt().mean([1, 2, 3], keepdim=True)
            y = y.repeat(G, 1, H, W)
            x = torch.cat([x, y], 1)
        else:
            x = torch.cat([x, torch.zeros(N, 1, H, W, device=x.device, dtype=x.dtype)], 1)
        x = self.conv_out(x)
        return self.out(self.fc(x.flatten(1)))
