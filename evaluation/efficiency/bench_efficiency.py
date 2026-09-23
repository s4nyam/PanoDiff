"""Measure inference cost and memory footprint of both PanoDiff-SR stages
(reviewer R2.6).

Training wall-clock is taken from the original training logs and is not
re-measured here. What this script contributes is the part the logs do not
record: parameter counts, per-image sampling/upscaling time, and peak device
memory for both inference and a training step.
"""
import argparse
import json
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SD = os.path.join(ROOT, "project-files/PanoDiff/Project/sd")
SR = os.path.join(ROOT, "project-files/PanoDiff/Project/sr/training")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=250)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default=os.path.join(HERE, "efficiency.json"))
args = ap.parse_args()

dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "ddim_steps": args.steps}


def sync():
    torch.cuda.synchronize()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 2**20


# --------------------------------------------------------------- PanoDiff --
sys.path.insert(0, SD)
from simple_diffusion.model import UNet                      # noqa: E402
from simple_diffusion.scheduler import DDIMScheduler         # noqa: E402

unet = UNet(3, image_size=(128, 256), hidden_dims=[64, 128, 256, 512]).to(dev).eval()
res["panodiff_params"] = sum(p.numel() for p in unet.parameters())
print(f"PanoDiff params: {res['panodiff_params']:,}")

sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")


@torch.no_grad()
def sample(bs, steps):
    """Exactly the call generate.py makes, so the timing is of the production
    sampling path (eta=1.0, batch of 32 by default)."""
    return sched.generate(unet, batch_size=bs, generator=torch.manual_seed(0),
                          eta=1.0, num_inference_steps=steps, device=dev)


for bs in (1, 8):
    # NB: the released model uses the non-flash attention path, which
    # materialises a B x heads x N x N score matrix at the 64x128 stage. That is
    # ~1 GiB per sample here, so batches beyond ~16 exceed a single device.
    sample(bs, 5)                                  # warm-up
    sync(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(args.reps if bs == 1 else 1):
        sync(); t0 = time.perf_counter()
        sample(bs, args.steps)
        sync(); ts.append(time.perf_counter() - t0)
    t = min(ts)
    res[f"panodiff_infer_bs{bs}_s"] = round(t, 2)
    res[f"panodiff_infer_bs{bs}_s_per_img"] = round(t / bs, 3)
    res[f"panodiff_infer_bs{bs}_peak_mb"] = round(peak_mb(), 1)
    print(f"  PanoDiff bs={bs:2d}: {t:8.2f} s total  {t/bs:6.3f} s/img  "
          f"peak {peak_mb():7.1f} MiB")

# training-step memory (batch 4, as used in the reported run)
unet.train()
opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)
torch.cuda.reset_peak_memory_stats()
x = torch.randn(4, 3, 128, 256, device=dev)
t = torch.randint(0, 1000, (4,), device=dev)
loss = torch.nn.functional.l1_loss(unet(x, t)["sample"], torch.randn_like(x))
loss.backward(); opt.step(); opt.zero_grad(); sync()
res["panodiff_train_bs4_peak_mb"] = round(peak_mb(), 1)
print(f"  PanoDiff train bs=4 peak: {peak_mb():.1f} MiB")
del unet, opt, loss, x, t
torch.cuda.empty_cache()

# --------------------------------------------------------------------- HAT --
# hat_arch pulls two trivial helpers from basicsr; stub them so the module can
# be imported without the full framework.
reg = types.ModuleType("basicsr.utils.registry")
reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
au = types.ModuleType("basicsr.archs.arch_util")
au.to_2tuple = lambda x: x if isinstance(x, tuple) else (x, x)
au.trunc_normal_ = torch.nn.init.trunc_normal_
for name, mod in [("basicsr", types.ModuleType("basicsr")),
                  ("basicsr.utils", types.ModuleType("basicsr.utils")),
                  ("basicsr.archs", types.ModuleType("basicsr.archs")),
                  ("basicsr.utils.registry", reg), ("basicsr.archs.arch_util", au)]:
    sys.modules[name] = mod

au.scandir = lambda *a, **k: []
sys.modules["basicsr.utils"].scandir = au.scandir

# Load hat_arch straight from its file: importing it as part of the `hat`
# package would execute an __init__ that pulls in the whole basicsr framework.
import importlib.util                                         # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "hat_arch", os.path.join(SR, "hat", "archs", "hat_arch.py"))
_hat_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hat_mod)
HAT = _hat_mod.HAT

hat = HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
          squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
          depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
          upsampler="pixelshuffle", resi_connection="1conv").to(dev).eval()
res["hat_params"] = sum(p.numel() for p in hat.parameters())
print(f"HAT params: {res['hat_params']:,}")

ckpt = os.path.join(ROOT, "project-files/PanoDiff/Project/sr/experiments/"
                    "train_Real_HAT_GAN_SRx4_finetune_from_mse_model/models/"
                    "net_g_400000.pth")
if os.path.exists(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("params_ema", sd.get("params", sd))
    missing, unexpected = hat.load_state_dict(sd, strict=False)
    print(f"  loaded {os.path.basename(ckpt)} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    res["hat_ckpt"] = os.path.basename(ckpt)


@torch.no_grad()
def upscale(bs):
    return hat(torch.randn(bs, 3, 128, 256, device=dev))


upscale(1)                                        # warm-up
sync(); torch.cuda.reset_peak_memory_stats()
ts = []
for _ in range(args.reps):
    sync(); t0 = time.perf_counter()
    y = upscale(1)
    sync(); ts.append(time.perf_counter() - t0)
t = min(ts)
res["hat_infer_bs1_s_per_img"] = round(t, 3)
res["hat_infer_bs1_peak_mb"] = round(peak_mb(), 1)
res["hat_output_shape"] = list(y.shape)
print(f"  HAT bs=1: {t:.3f} s/img  peak {peak_mb():.1f} MiB  out {tuple(y.shape)}")

json.dump(res, open(args.out, "w"), indent=2)
print("\nwrote", args.out)
print(json.dumps(res, indent=2))
