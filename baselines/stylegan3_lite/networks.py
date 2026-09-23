"""StyleGAN3-T generator (Karras et al., NeurIPS 2021) at native 512x1024 on LUMI.

The layers are NVIDIA's own code: MappingNetwork, SynthesisInput and SynthesisLayer
are imported unchanged from repos/stylegan3/training/networks_stylegan3.py. Two
things are replaced, and nothing else:

  * filtered_lrelu. NVIDIA's fused CUDA kernel cannot build on ROCm, and the repo's
    reference fallback runs the full-rate low-pass filter and then discards three of
    every four samples when downsampling. `filtered_lrelu_fast` computes the same
    function (bias -> zero-insert upsample -> FIR -> leaky ReLU x gain -> clamp ->
    FIR -> decimate) with the decimation folded into a strided separable filter and
    the zero-insertion upsampling done as a strided transposed convolution, so the
    zero-stuffed array is never built. tools/test_sg3.py checks the whole generator,
    forward and backward, against the reference path.

  * SynthesisNetwork, only to allow a non-square canvas. The official network places
    the image on a unit square sampled at `img_resolution` samples per unit. Here the
    canvas is `aspect` = (2, 1) units, so every layer keeps the official sampling
    rates, cutoffs, filters and channel counts of a 512 network, and only the
    spatial size of each feature map becomes [2*s + 2*margin, s + 2*margin]. The
    Fourier input already takes a [width, height] size and places its grid in the
    same units, so the equivariance construction is untouched.

The discriminator is the StyleGAN2 residual discriminator, as in NVIDIA's release
(`--cfg=stylegan3-t` uses training.networks_stylegan2.Discriminator); it is shared
with stylegan2_lite.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os, sys, types
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("SG_FORCE_REF", "1")          # never try to build the CUDA ops
REPO = _os.environ.get("STYLEGAN3_REPO", WORK + "/new-baselines-generation/repos/stylegan3")  # clone of NVlabs/stylegan3
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from training import networks_stylegan3 as nv                      # noqa: E402
from torch_utils.ops import filtered_lrelu as nv_flrelu            # noqa: E402


# ----------------------------------------------------------------------------- op

_FIR_MODE = os.environ.get("SG3_FIR", "group")      # 'group' (depthwise) or 'reshape'
_UP_MODE = os.environ.get("SG3_UP", "tconv")        # 'tconv' (polyphase) or 'zero'

def _fir_sep(x, f, stride, axis):
    """Depthwise 1-D FIR along `axis` (3 = width, 2 = height), 'valid' padding.
    `f` is already flipped/scaled; a stride > 1 decimates in the same pass."""
    N, C, H, W = x.shape
    k = f.numel()
    if _FIR_MODE == "reshape":                       # channels folded into the batch
        y = x.reshape(N * C, 1, H, W)
        if axis == 3:
            y = F.conv2d(y, f.view(1, 1, 1, k), stride=(1, stride))
        else:
            y = F.conv2d(y, f.view(1, 1, k, 1), stride=(stride, 1))
        return y.reshape(N, C, y.shape[2], y.shape[3])
    if axis == 3:
        w = f.view(1, 1, 1, k).expand(C, 1, 1, k)
        return F.conv2d(x, w, stride=(1, stride), groups=C)
    w = f.view(1, 1, k, 1).expand(C, 1, k, 1)
    return F.conv2d(x, w, stride=(stride, 1), groups=C)


def _up_sep(x, w, up, p0, p1, axis):
    """Zero-insert by `up`, pad by (p0, p1), 'valid' FIR -- along one axis -- computed
    as a transposed convolution. With T = conv_transpose(x, w, stride=up), the
    reference output is out[j] = T[j + K - 1 - p0] (zero outside T)."""
    N, C, H, W = x.shape
    K = w.numel()
    L = (W if axis == 3 else H)
    n_out = L * up + p0 + p1 - K + 1
    if axis == 3:
        T = F.conv_transpose2d(x, w.view(1, 1, 1, K).expand(C, 1, 1, K), stride=(1, up), groups=C)
    else:
        T = F.conv_transpose2d(x, w.view(1, 1, K, 1).expand(C, 1, K, 1), stride=(up, 1), groups=C)
    start = K - 1 - p0
    lo, hi = max(-start, 0), max(start + n_out - T.shape[axis], 0)
    if lo or hi:
        T = F.pad(T, [lo, hi, 0, 0] if axis == 3 else [0, 0, lo, hi])
    start += lo
    return T.narrow(axis, start, n_out)


def filtered_lrelu_fast(x, fu=None, fd=None, b=None, up=1, down=1, padding=0,
                        gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl=None):
    """Same function as torch_utils.ops.filtered_lrelu._filtered_lrelu_ref for the
    separable (1-D) filters that StyleGAN3-T uses."""
    assert fu is None or fu.ndim == 1
    assert fd is None or fd.ndim == 1
    px0, px1, py0, py1 = nv_flrelu._parse_padding(padding)
    N, C, H, W = x.shape
    if b is not None:
        x = x + b.view(1, -1, 1, 1).to(x.dtype)
    # upsample: zero insertion, pad/crop, FIR with gain `up` per axis (up**2 total)
    if fu is not None and up > 1 and _UP_MODE == "tconv":
        # polyphase form: a strided transposed convolution per axis never builds the
        # zero-stuffed array, then the crop/pad reproduces the reference's padding
        f = fu.to(x.dtype) * up
        if flip_filter:
            f = f.flip(0)
        x = _up_sep(x, f, up, px0, px1, axis=3)
        x = _up_sep(x, f, up, py0, py1, axis=2)
    else:
        if up > 1:
            x = x.reshape(N, C, H, 1, W, 1)
            x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
            x = x.reshape(N, C, H * up, W * up)
        x = F.pad(x, [max(px0, 0), max(px1, 0), max(py0, 0), max(py1, 0)])
        x = x[:, :, max(-py0, 0): x.shape[2] - max(-py1, 0), max(-px0, 0): x.shape[3] - max(-px1, 0)]
        if fu is not None:
            f = fu.to(x.dtype) * up
            if not flip_filter:
                f = f.flip(0)
            x = _fir_sep(x, f, 1, 3)
            x = _fir_sep(x, f, 1, 2)
    # bias already added; leaky ReLU, gain, clamp
    x = F.leaky_relu(x, slope) if slope != 1 else x
    if gain != 1:
        x = x * gain
    if clamp is not None and clamp >= 0:
        x = x.clamp(-clamp, clamp)
    # downsample: FIR and decimation in one strided pass per axis
    if fd is not None:
        f = fd.to(x.dtype)
        if not flip_filter:
            f = f.flip(0)
        x = _fir_sep(x, f, down, 3)
        x = _fir_sep(x, f, down, 2)
    elif down > 1:
        x = x[:, :, ::down, ::down]
    return x


# route NVIDIA's SynthesisLayer through the fast op (it calls filtered_lrelu.filtered_lrelu)
_shim = types.SimpleNamespace(filtered_lrelu=filtered_lrelu_fast)


def use_fast_op(enable=True):
    nv.filtered_lrelu = _shim if enable else nv_flrelu


use_fast_op(True)


# ----------------------------------------------------------------------------- network

class SynthesisNetwork(torch.nn.Module):
    """NVIDIA's SynthesisNetwork with a [width, height] = aspect canvas."""
    def __init__(self, w_dim=512, img_resolution=512, aspect=(2, 1), img_channels=3,
                 channel_base=32768, channel_max=512, num_layers=14, num_critical=2,
                 first_cutoff=2, first_stopband=2 ** 2.1, last_stopband_rel=2 ** 0.3,
                 margin_size=10, output_scale=0.25, **layer_kwargs):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.aspect = np.asarray(aspect)
        self.img_size = (self.aspect * img_resolution).astype(int)        # [W, H]
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.margin_size = margin_size
        self.output_scale = output_scale

        # --- identical to networks_stylegan3.SynthesisNetwork ---
        last_cutoff = img_resolution / 2
        last_stopband = last_cutoff * last_stopband_rel
        exponents = np.minimum(np.arange(num_layers + 1) / (num_layers - num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, img_resolution))))
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = img_channels
        # --- only change: 2-D sizes on the aspect canvas ---
        sizes = [(self.aspect * s + margin_size * 2).astype(int) for s in sampling_rates]
        sizes[-2] = sizes[-1] = self.img_size

        self.input = nv.SynthesisInput(w_dim=w_dim, channels=int(channels[0]), size=sizes[0],
                                       sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(num_layers + 1):
            prev = max(idx - 1, 0)
            layer = nv.SynthesisLayer(
                w_dim=w_dim, is_torgb=(idx == num_layers),
                is_critically_sampled=(idx >= num_layers - num_critical), use_fp16=False,
                in_channels=int(channels[prev]), out_channels=int(channels[idx]),
                in_size=sizes[prev], out_size=sizes[idx],
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f"L{idx}_{layer.out_size[0]}x{layer.out_size[1]}_{layer.out_channels}"
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, update_emas=False):
        ws = ws.to(torch.float32).unbind(dim=1)
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, update_emas=update_emas)
        if self.output_scale != 1:
            x = x * self.output_scale
        return x.to(torch.float32)


class Generator(torch.nn.Module):
    """StyleGAN3-T: 2-layer mapping network, alias-free synthesis network.
    `img_resolution` = (H, W) of the output, H the per-unit sampling rate."""
    def __init__(self, z_dim=512, w_dim=512, img_resolution=(512, 1024), img_channels=3,
                 channel_base=32768, channel_max=512, magnitude_ema_beta=0.999, map_layers=2):
        super().__init__()
        H, W = img_resolution
        assert W % H == 0
        self.z_dim, self.w_dim = z_dim, w_dim
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=H, aspect=(W // H, 1),
                                          img_channels=img_channels, channel_base=channel_base,
                                          channel_max=channel_max,
                                          magnitude_ema_beta=magnitude_ema_beta)
        self.num_ws = self.synthesis.num_ws
        self.mapping = nv.MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim,
                                         num_ws=self.num_ws, num_layers=map_layers)

    def forward(self, z, truncation_psi=1.0, update_emas=False):
        ws = self.mapping(z, None, truncation_psi=truncation_psi, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas)
