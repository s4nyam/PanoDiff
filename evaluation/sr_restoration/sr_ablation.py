"""SR-stage ablations for reviewer R2.4.

Evaluates the variants of the super-resolution stage that can be compared
exactly, i.e. without retraining, because every BasicSR checkpoint stores both
the raw and the EMA weights and checkpoints were written every 10k iterations:

    bicubic                 no learned SR at all (floor)
    pretrained              Real_HAT_GAN_SRx4 as released, no fine-tuning
    ft-10k .. ft-400k       our fine-tuning at increasing budget (EMA weights)
    ft-400k-noema           the same run read from 'params' instead of 'params_ema'

Protocol: each real PR at 1024x512 is bicubic-downsampled by 4 to 256x128,
passed through the variant, and compared with the original. PSNR and SSIM are
computed on the luminance channel, LPIPS with the AlexNet backbone.

Note on leakage, which the write-up states as well: all 7243 real PRs were used
to fine-tune the SR stage, so no held-out real set exists. Comparisons between
fine-tuned variants (EMA vs raw, budget) are unaffected, since every variant
comes from the same run and saw exactly the same images. The pretrained and
bicubic rows are the ones that did not see this data, so the margin over them is
optimistic and is reported as an upper bound rather than a generalisation gap.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

P = WORK + "/project-files/PanoDiff/Project"
HAT_ARCH = P + "/sr/training/hat/archs/hat_arch.py"
FT_DIR = P + "/sr/experiments/train_Real_HAT_GAN_SRx4_finetune_from_mse_model/models"
PRETRAINED = P + "/sr/training/experiments/pretrained_models/Real_HAT_GAN_SRx4.pth"
HR_DIR = P + "/sr/training/datasets/images/HR"

HAT_KW = dict(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3,
              squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
              depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
              upsampler='pixelshuffle', resi_connection='1conv')


def load_hat():
    """Import hat_arch.py directly.

    basicsr is not installed in the container, and hat_arch.py needs only three
    things from it: the ARCH_REGISTRY decorator, to_2tuple and trunc_normal_.
    The last two are the standard timm/torch helpers, so stubbing the two
    modules gives the architecture unchanged rather than a reimplementation.
    """
    import collections.abc
    from itertools import repeat
    import torch.nn.init as init

    def to_2tuple(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, 2))

    pkg = type(sys)("basicsr"); pkg.__path__ = []
    sys.modules["basicsr"] = pkg

    class _R:
        def register(self, *a, **k):
            return (lambda c: c) if not a else a[0]

    reg = type(sys)("basicsr.utils.registry"); reg.ARCH_REGISTRY = _R()
    utils = type(sys)("basicsr.utils"); utils.__path__ = []; utils.registry = reg
    archs = type(sys)("basicsr.archs"); archs.__path__ = []
    arch_util = type(sys)("basicsr.archs.arch_util")
    arch_util.to_2tuple = to_2tuple
    arch_util.trunc_normal_ = init.trunc_normal_
    archs.arch_util = arch_util
    for k, v in [("basicsr.utils", utils), ("basicsr.utils.registry", reg),
                 ("basicsr.archs", archs), ("basicsr.archs.arch_util", arch_util)]:
        sys.modules[k] = v

    spec = importlib.util.spec_from_file_location("hat_arch", HAT_ARCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.HAT


def to_tensor(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def to_uint8(t):
    a = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (a * 255.0 + 0.5).astype(np.uint8)


def luma(a):
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--save-images", default="")
    ap.add_argument("--save-names", default="")
    ap.add_argument("--save-first", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAT = load_hat()

    names = sorted(os.listdir(HR_DIR))
    rng = np.random.RandomState(1234)
    sel = [names[i] for i in rng.choice(len(names), args.n, replace=False)]

    variants = {
        "bicubic":       ("bicubic", None),
        "pretrained":    (PRETRAINED, "params_ema"),
        "ft-10k":        (FT_DIR + "/net_g_10000.pth", "params_ema"),
        "ft-50k":        (FT_DIR + "/net_g_50000.pth", "params_ema"),
        "ft-100k":       (FT_DIR + "/net_g_100000.pth", "params_ema"),
        "ft-200k":       (FT_DIR + "/net_g_200000.pth", "params_ema"),
        "ft-400k":       (FT_DIR + "/net_g_400000.pth", "params_ema"),
        "ft-400k-noema": (FT_DIR + "/net_g_400000.pth", "params"),
    }
    if args.variants != "all":
        keep = args.variants.split(",")
        variants = {k: v for k, v in variants.items() if k in keep}

    import lpips
    lp = lpips.LPIPS(net="alex").to(dev)

    save_names = args.save_names.split(",") if args.save_names else []
    results = {}

    for vname, (ckpt, key) in variants.items():
        net = None
        if ckpt != "bicubic":
            net = HAT(**HAT_KW)
            sd = torch.load(ckpt, map_location="cpu")
            net.load_state_dict(sd[key] if key in sd else sd, strict=True)
            net.eval().to(dev)

        ps, ss, lps = [], [], []
        for i, fn in enumerate(sel):
            hr = Image.open(os.path.join(HR_DIR, fn)).convert("RGB")
            W, H = hr.size
            lr = hr.resize((W // 4, H // 4), Image.BICUBIC)

            if net is None:
                sr_img = lr.resize((W, H), Image.BICUBIC)
                out = to_tensor(sr_img).to(dev)
            else:
                with torch.no_grad():
                    out = net(to_tensor(lr).to(dev)).clamp(0, 1)

            gt = np.asarray(hr, dtype=np.float32) / 255.0
            pr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pr = pr[:gt.shape[0], :gt.shape[1]]

            ps.append(psnr_fn(luma(gt), luma(pr), data_range=1.0))
            ss.append(ssim_fn(luma(gt), luma(pr), data_range=1.0))
            with torch.no_grad():
                g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(dev) * 2 - 1
                p = out * 2 - 1
                lps.append(float(lp(g, p[:, :, :g.shape[2], :g.shape[3]]).item()))

            if args.save_images and (fn in save_names or i < args.save_first):
                os.makedirs(args.save_images, exist_ok=True)
                Image.fromarray(to_uint8(out)).save(
                    os.path.join(args.save_images, "%s__%s.png" % (fn[:-4], vname)))
                if vname == "bicubic":
                    hr.save(os.path.join(args.save_images, "%s__gt.png" % fn[:-4]))
                    lr.save(os.path.join(args.save_images, "%s__lr.png" % fn[:-4]))

            if (i + 1) % 25 == 0:
                print("  %-14s %3d/%d  PSNR %.3f" % (vname, i + 1, len(sel), np.mean(ps)),
                      flush=True)

        results[vname] = dict(psnr=float(np.mean(ps)), psnr_sd=float(np.std(ps)),
                              ssim=float(np.mean(ss)), ssim_sd=float(np.std(ss)),
                              lpips=float(np.mean(lps)), lpips_sd=float(np.std(lps)),
                              n=len(sel))
        print("%-16s PSNR %.3f  SSIM %.4f  LPIPS %.4f"
              % (vname, results[vname]["psnr"], results[vname]["ssim"],
                 results[vname]["lpips"]), flush=True)
        del net
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
