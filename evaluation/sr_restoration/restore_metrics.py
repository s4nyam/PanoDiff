"""Restoration of all 7243 real PRs by the two upscalers of Section 4.5.1 (replaces the 300-image table).

Set 1 of Tables 8-9 is set 10 (real PRs, 1024x512) downsampled fourfold by area averaging
(cv2.INTER_AREA, sr/training/datasets/images/create_lr.py; identical to LQ4, the SR training
inputs). Each row applies one checkpoint to set 1 and compares the result with set 10:

    hat_ft   HAT-SR as used in 4.5.1: net_g_400000, params_ema (reproduces set 4 from set 2)
    swin_ft  SwinIR fine-tuned on the PR pairs, as used in 4.5.1 (final_model.pth); these are
             swinir-ft-eval/out/9ft-TrainSwinIRft, made from set 1 by infer_swinir_ft.py
    hat_pre  HAT, released weights (Real_HAT_GAN_SRx4, params_ema)
    swin_pre SwinIR, released weights (003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN), with the
             preprocessing of infer_swinir_ft.py, which reproduces the paper's set 6

The paper's set 8 (8-TrainHATSR) is NOT used: the HAT-SR checkpoint does not reproduce it
(mean abs diff ~4.5 grey levels), nor do the released weights, so its provenance is unclear.

Metrics are those of the earlier table (analysis/sr_ablation.py): PSNR and SSIM on luminance
(skimage, data_range 1) and LPIPS (AlexNet, inputs in [-1, 1]), each on the full image, computed
from the saved 8-bit outputs so that every row is scored the same way.

Rank 0 first records how the checkpoints relate to the paper's sets (out/validate_hat.json):
HAT-SR on set 2 against set 4, HAT-SR on set 1 against set 8, released SwinIR on set 1 against
set 9.

Sharded over SLURM tasks: rank r of W takes names[r::W] and writes out/parts/rank_XXX.csv.
Existing outputs are reused, so the job is resumable.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import csv, json, os, sys
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

A = WORK + "/new-files/analysis"
sys.path.insert(0, A)
from sr_ablation import load_hat, HAT_KW, to_tensor, to_uint8, luma, FT_DIR, PRETRAINED  # noqa: E402
sys.path.insert(0, WORK + "/swinir-ft-eval/code")
import infer_swinir_ft as swi  # noqa: E402

P = WORK
LEG = f"{P}/project-files/PanoDiff/Syn-Calc-FID&IS"
OUT = f"{P}/sr-restore-7243/out"
HR = f"{LEG}/10-TrainDataInHR"
LR = f"{LEG}/1-TrainingDataInLR"
HAT_PRE_DIR = f"{OUT}/8pre-TrainHATpre"
HAT_FT_DIR = f"{OUT}/8ft-TrainHATSR"
SWIN_PRE_DIR = f"{OUT}/9pre-TrainSwinIRpre"
ROWS = {
    "hat_pre":  lambda n: f"{HAT_PRE_DIR}/{n[:-4]}__SR.png",
    "swin_pre": lambda n: f"{SWIN_PRE_DIR}/{n[:-4]}_SwinIR.png",
    "hat_ft":   lambda n: f"{HAT_FT_DIR}/{n[:-4]}__SR.png",
    "swin_ft":  lambda n: f"{P}/swinir-ft-eval/out/9ft-TrainSwinIRft/{n[:-4]}_SwinIR.png",
}


def hat(ckpt, dev):
    net = load_hat()(**HAT_KW)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["params_ema"], strict=True)
    return net.eval().to(dev)


def upscale(net, path, dev):
    with torch.no_grad():
        return to_uint8(net(to_tensor(Image.open(path)).to(dev)).clamp(0, 1))


def swin_up(net, path, dev):
    return swi.upscale(net, swi.load_lr(path).unsqueeze(0), dev)[0]


def validate(dev, names):
    """How the checkpoints relate to the paper's sets 4, 8 and 9."""
    net = hat(FT_DIR + "/net_g_400000.pth", dev)
    res = {}
    gen = sorted(os.listdir(f"{LEG}/2-DiffLR"))
    for tag, items in [("set8", [(f"{LR}/{n}", f"{LEG}/8-TrainHATSR/{n[:-4]}__SR.png") for n in names[::450][:16]]),
                       ("set4", [(f"{LEG}/2-DiffLR/{n}", f"{LEG}/4-DiffHATSR/{n[:-4]}__SR.png") for n in gen[::900][:8]])]:
        d = [np.abs(upscale(net, a, dev).astype(int) - np.asarray(Image.open(b).convert("RGB")).astype(int))
             for a, b in items]
        res[tag] = dict(n=len(d), max_abs_diff=int(max(x.max() for x in d)),
                        mean_abs_diff=float(np.mean([x.mean() for x in d])),
                        identical_images=int(sum(x.max() == 0 for x in d)))
        print("validate", tag, res[tag], flush=True)
    del net
    net = swi.build(swi.SW_PRE, dev)
    d = [np.abs(swin_up(net, f"{LR}/{n}", dev).astype(int)
                - np.asarray(Image.open(f"{LEG}/9-TrainSwinIR/{n[:-4]}_SwinIR.png").convert("RGB")).astype(int))
         for n in names[::450][:16]]
    res["set9"] = dict(n=len(d), max_abs_diff=int(max(x.max() for x in d)),
                       mean_abs_diff=float(np.mean([x.mean() for x in d])),
                       identical_images=int(sum(x.max() == 0 for x in d)))
    print("validate set9", res["set9"], flush=True)
    json.dump(res, open(f"{OUT}/validate_hat.json", "w"), indent=1)
    del net


def main():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in (HAT_PRE_DIR, HAT_FT_DIR, SWIN_PRE_DIR, f"{OUT}/parts"):
        os.makedirs(d, exist_ok=True)
    names = sorted(n for n in os.listdir(HR) if n.endswith(".png"))
    assert len(names) == 7243, len(names)
    if rank == 0:
        validate(dev, names)
    mine = names[rank::world]

    for row, make in [("hat_pre", lambda: (hat(PRETRAINED, dev), upscale)),
                      ("hat_ft", lambda: (hat(FT_DIR + "/net_g_400000.pth", dev), upscale)),
                      ("swin_pre", lambda: (swi.build(swi.SW_PRE, dev), swin_up))]:
        todo = [n for n in mine if not os.path.exists(ROWS[row](n))]
        if todo:
            net, fn = make()
            for n in todo:
                o = ROWS[row](n); tmp = os.path.join(os.path.dirname(o), "." + os.path.basename(o))
                Image.fromarray(fn(net, f"{LR}/{n}", dev)).save(tmp, format="PNG")
                os.replace(tmp, o)
            del net
        print(f"rank {rank}: {row} done ({len(todo)} made)", flush=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    with open(f"{OUT}/parts/rank_{rank:03d}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "row", "psnr", "ssim", "lpips"])
        for i, n in enumerate(mine):
            gt = np.asarray(Image.open(f"{HR}/{n}").convert("RGB"), dtype=np.float32) / 255.0
            g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(dev) * 2 - 1
            for row, path in ROWS.items():
                pr = np.asarray(Image.open(path(n)).convert("RGB"), dtype=np.float32) / 255.0
                assert pr.shape == gt.shape, (row, n, pr.shape)
                p = torch.from_numpy(pr).permute(2, 0, 1).unsqueeze(0).to(dev) * 2 - 1
                with torch.no_grad():
                    l = float(lp(g, p).item())
                w.writerow([n, row, psnr_fn(luma(gt), luma(pr), data_range=1.0),
                            ssim_fn(luma(gt), luma(pr), data_range=1.0), l])
            if (i + 1) % 50 == 0:
                print(f"rank {rank}: scored {i + 1}/{len(mine)}", flush=True)
    print(f"rank {rank}: done", flush=True)


if __name__ == "__main__":
    main()
