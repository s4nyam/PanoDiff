"""Table 10 on held-out real PRs: the four checkpoints of restore_metrics.py applied to the 185
DENTEX validation/test radiographs of prep_heldout.py (LR -> 1024x512), scored against HR with
the same PSNR/SSIM (luminance) and LPIPS (AlexNet) as before.

Rank 0 first checks that the fine-tuned SwinIR path reproduces the paper's 9ft set (made by
infer_swinir_ft.py from set 1); the HAT-SR checkpoint was checked against set 4 in the 7243 run.
Sharded over SLURM tasks; writes heldout/parts/rank_XXX.csv and heldout/validate_swinft.json.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import csv, json, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restore_metrics as rm  # noqa: E402  (HAT/SwinIR loaders, upscalers, metric helpers)
from restore_metrics import hat, upscale, swin_up, swi, luma, psnr_fn, ssim_fn, FT_DIR, PRETRAINED  # noqa: E402

H = os.environ.get("HELD_DIR",
                   WORK + "/sr-restore-7243/heldout")
MAKERS = {
    "hat_pre":  lambda dev: (hat(PRETRAINED, dev), upscale),
    "swin_pre": lambda dev: (swi.build(swi.SW_PRE, dev), swin_up),
    "hat_ft":   lambda dev: (hat(FT_DIR + "/net_g_400000.pth", dev), upscale),
    "swin_ft":  lambda dev: (swi.build(swi.SW_FT, dev), swin_up),
}


def main():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = json.load(open(f"{H}/heldout_list.json"))["names"]
    for r in MAKERS:
        os.makedirs(f"{H}/out/{r}", exist_ok=True)
    os.makedirs(f"{H}/parts", exist_ok=True)

    if rank == 0:
        net = swi.build(swi.SW_FT, dev)
        d = []
        for n in sorted(os.listdir(rm.LR))[::1800][:4]:
            ref = np.asarray(Image.open(rm.ROWS["swin_ft"](n)).convert("RGB")).astype(int)
            d.append(np.abs(swin_up(net, f"{rm.LR}/{n}", dev).astype(int) - ref))
        v = dict(n=len(d), max_abs_diff=int(max(x.max() for x in d)), mean_abs_diff=float(np.mean([x.mean() for x in d])))
        json.dump(v, open(f"{H}/validate_swinft.json", "w"), indent=1)
        print("validate swin_ft vs 9ft", v, flush=True)
        del net

    mine = names[rank::world]
    for row, make in MAKERS.items():
        net, fn = make(dev)
        for n in mine:
            Image.fromarray(fn(net, f"{H}/LR/{n}", dev)).save(f"{H}/out/{row}/{n}")
        del net
        print(f"rank {rank}: {row} done ({len(mine)})", flush=True)

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    with open(f"{H}/parts/rank_{rank:03d}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "row", "psnr", "ssim", "lpips"])
        for n in mine:
            gt = np.asarray(Image.open(f"{H}/HR/{n}").convert("RGB"), dtype=np.float32) / 255.0
            g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(dev) * 2 - 1
            for row in MAKERS:
                pr = np.asarray(Image.open(f"{H}/out/{row}/{n}").convert("RGB"), dtype=np.float32) / 255.0
                assert pr.shape == gt.shape, (row, n, pr.shape)
                p = torch.from_numpy(pr).permute(2, 0, 1).unsqueeze(0).to(dev) * 2 - 1
                with torch.no_grad():
                    l = float(lp(g, p).item())
                w.writerow([n, row, psnr_fn(luma(gt), luma(pr), data_range=1.0),
                            ssim_fn(luma(gt), luma(pr), data_range=1.0), l])
    print(f"rank {rank}: done", flush=True)


if __name__ == "__main__":
    main()
