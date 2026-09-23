"""Re-run the SwinIR arms of Section 4.5.1 with the PR-fine-tuned SwinIR.

In the paper's Tables 8-9 the SwinIR rows (6 LRPD+SwinIR, 7 LRGAN+SwinIR, 9 LRGT+SwinIR) were
made with SwinIR's released natural-image weights, while HAT-SR had been fine-tuned on the
7243 PR pairs. This produces the same three sets with the SwinIR fine-tuned on those pairs
(recovered checkpoint final_model.pth = 10 epochs, batch 8, 9054 updates, L1 loss), so that
both upscalers in 4.5.1 have seen the PR data.

Only the weights change. Preprocessing replicates main_test_swinir.py, which made the original
outputs: RGB in [0,1], flip-padding to a multiple of the window (8), 4x forward pass, crop to
4x the input, clamp, round to uint8. Architecture, strict loading and the timm stub are those
of new-files/analysis/swinir_r4maj2/swinir_matched.py, which produced Table 10.

Sharded over SLURM tasks: rank r of W processes items[r::W]; existing outputs are skipped, so
the job is resumable. Rank 0 first regenerates --validate images with the PRE-TRAINED weights
and compares them pixel-wise with the paper's existing 6-DiffSwinIR files; the result goes to
out/validate.json. If that check fails, the preprocessing differs and the run is not valid.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, collections.abc, importlib.util, json, os, sys, time
from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image

P = WORK
LEG = f"{P}/project-files/PanoDiff/Syn-Calc-FID&IS"
SW = f"{P}/my-old-machine-just-got-recovered/swinir"
SW_PRE = f"{SW}/experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
SW_FT = f"{SW}/checkpoints/finetune/final_model.pth"
STOCK = f"{P}/new-files/analysis/swinir_r4maj2/network_swinir_stock.py"
OUT = f"{P}/swinir-ft-eval/out"
SETS = [  # (input LR folder, output folder) -- output names mirror the paper's 6/7/9 folders
    ("2-DiffLR",           "6ft-DiffSwinIRft"),
    ("3-GANsLR",           "7ft-GANsSwinIRft"),
    ("1-TrainingDataInLR", "9ft-TrainSwinIRft"),
]
SW_KW = dict(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.,
             depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
             upsampler='nearest+conv', resi_connection='1conv')
WIN, SCALE = 8, 4


def swinir_class():
    """network_swinir needs DropPath, to_2tuple, trunc_normal_ from timm (absent in the container)."""
    def to_2tuple(x):
        return tuple(x) if isinstance(x, collections.abc.Iterable) and not isinstance(x, str) else tuple(repeat(x, 2))

    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__(); self.drop_prob = drop_prob
        def forward(self, x):
            if not self.drop_prob or not self.training:
                return x
            keep = 1 - self.drop_prob
            mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
            return x.div(keep) * mask.floor_()

    timm = type(sys)("timm"); timm.__path__ = []
    models = type(sys)("timm.models"); models.__path__ = []
    layers = type(sys)("timm.models.layers")
    layers.DropPath, layers.to_2tuple, layers.trunc_normal_ = DropPath, to_2tuple, init.trunc_normal_
    models.layers = layers; timm.models = models
    for k, v in (("timm", timm), ("timm.models", models), ("timm.models.layers", layers)):
        sys.modules.setdefault(k, v)
    spec = importlib.util.spec_from_file_location("network_swinir_stock", STOCK)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.SwinIR


def build(ckpt, dev):
    net = swinir_class()(**SW_KW)
    sd = torch.load(ckpt, map_location="cpu")
    for k in ("params_ema", "params", "model_state_dict", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]; break
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.eval().to(dev)


def load_lr(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0      # HWC RGB
    return torch.from_numpy(a.transpose(2, 0, 1).copy())                            # CHW


@torch.no_grad()
def upscale(net, batch, dev):
    """batch: [B,3,h,w] in [0,1] -> uint8 [B,h*4,w*4,3], exactly as main_test_swinir.py."""
    x = batch.to(dev)
    _, _, h, w = x.shape
    hp = (h // WIN + 1) * WIN - h
    wp = (w // WIN + 1) * WIN - w
    x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp, :]
    x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
    y = net(x)[..., :h * SCALE, :w * SCALE].clamp_(0, 1)
    return (y.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


def out_name(fname):
    return os.path.splitext(fname)[0] + "_SwinIR.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--validate", type=int, default=16)
    a = ap.parse_args()
    rank = int(os.environ.get("RANK_OVERRIDE", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_OVERRIDE", os.environ.get("SLURM_NTASKS", 1)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    if rank == 0 and a.validate:
        # the paper's own SwinIR outputs must be reproduced by this code with the released weights
        pre = build(SW_PRE, dev)
        names = sorted(f for f in os.listdir(f"{LEG}/2-DiffLR") if f.endswith(".png"))[:a.validate]
        x = torch.stack([load_lr(f"{LEG}/2-DiffLR/{n}") for n in names])
        got = upscale(pre, x, dev)
        diffs = []
        for n, g in zip(names, got):
            ref = np.asarray(Image.open(f"{LEG}/6-DiffSwinIR/{out_name(n)}").convert("RGB"))
            diffs.append((int(np.abs(g.astype(int) - ref.astype(int)).max()),
                          float(np.abs(g.astype(int) - ref.astype(int)).mean())))
        v = {"n": len(names), "max_abs_diff": max(d[0] for d in diffs),
             "mean_abs_diff": float(np.mean([d[1] for d in diffs])),
             "identical_images": sum(d[0] == 0 for d in diffs)}
        os.makedirs(OUT, exist_ok=True)
        json.dump(v, open(f"{OUT}/validate.json", "w"), indent=1)
        print("VALIDATE (pretrained weights vs paper's 6-DiffSwinIR):", v, flush=True)
        del pre; torch.cuda.empty_cache()

    net = build(SW_FT, dev)
    items = []
    for src, dst in SETS:
        os.makedirs(f"{OUT}/{dst}", exist_ok=True)
        items += [(src, dst, f) for f in sorted(os.listdir(f"{LEG}/{src}")) if f.endswith(".png")]
    mine = [it for it in items[rank::world] if not os.path.exists(f"{OUT}/{it[1]}/{out_name(it[2])}")]
    print(f"rank {rank}/{world}: {len(mine)} images to do", flush=True)
    t0 = time.time()
    for s in range(0, len(mine), a.batch):
        chunk = mine[s:s + a.batch]
        ys = upscale(net, torch.stack([load_lr(f"{LEG}/{src}/{f}") for src, _, f in chunk]), dev)
        for (src, dst, f), y in zip(chunk, ys):
            tmp = f"{OUT}/{dst}/.{out_name(f)}.tmp.png"
            Image.fromarray(y).save(tmp)
            os.replace(tmp, f"{OUT}/{dst}/{out_name(f)}")
        if (s // a.batch) % 25 == 0:
            done = s + len(chunk)
            print(f"rank {rank}: {done}/{len(mine)}  {done / max(time.time() - t0, 1e-6):.1f} img/s", flush=True)
    print(f"rank {rank}: DONE {len(mine)} images in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
