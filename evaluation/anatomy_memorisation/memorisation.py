"""Nearest-neighbour memorisation check.

FID here is computed against the same 7243 real PRs every model trained on, so it
rewards copying by construction: a generator that reproduced its training set
exactly would score FID ~0. Before reporting that StyleGAN2-ADA attains a better
FID than the paper's pipeline, we have to establish that it is not doing so by
reproducing training images.

Method. Every image is embedded with the SAME Inception network used for FID (so
the analysis lives in the space the metric is computed in), features are
L2-normalised, and for each generated image we find its nearest real training
image by cosine distance. The reference distribution is the leave-one-out
nearest-neighbour distance among the real images themselves: since all 7243 reals
were used in training there is no held-out real set, so "how close is a real PR to
its closest other real PR" is the only honest yardstick for how close is too
close. A generator that memorises produces NN distances BELOW that reference; a
generator that generalises produces distances at or above it.

Also writes, per arm, a side-by-side figure of the 8 generated images with the
smallest NN distance next to the real image they matched -- the qualitative
evidence a reviewer will want to see alongside the numbers.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import torchvision.transforms as T

B = WORK + "/new-baselines-generation"
REAL = os.path.join(B, "refs/real")
ARMS = [("sg2ada", "samples/sg2ada"), ("adm", "samples/adm"), ("ldm", "samples/ldm"),
        ("panodiff_hr", "samples/panodiff_hr"),
        ("PanoDiffSR_existing", "refs/PanoDiffSR_existing"),
        ("FastGAN_HATSR_existing", "refs/FastGAN_HATSR_existing"),
        ("sg3", "samples/sg3"), ("fastgan_lite64", "samples/fastgan_lite64")]
# ONLY_ARM scores one arm into memorisation_<arm>.json, leaving memorisation.json as reported
_only = os.environ.get("ONLY_ARM")
if _only:
    ARMS = [a for a in ARMS if a[0] == _only]


class Folder(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.paths = sorted(f for f in os.listdir(folder)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = T.ToTensor()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.t(Image.open(os.path.join(self.folder, self.paths[i])).convert("RGB"))


@torch.no_grad()
def features(folder, extractor, dev, batch=64):
    out = []
    for b in DataLoader(Folder(folder), batch_size=batch, num_workers=6, pin_memory=True):
        # torchmetrics' FID extractor expects uint8 in [0,255]
        f = extractor((b.to(dev) * 255).to(torch.uint8))
        out.append(f.float().cpu())
    return torch.cat(out)


def main():
    dev = torch.device("cuda:0")
    extractor = FrechetInceptionDistance(normalize=True).to(dev).inception.eval()

    fr = features(REAL, extractor, dev)
    fr = torch.nn.functional.normalize(fr, dim=1).to(dev)
    real_names = Folder(REAL).paths
    print(f"real features {tuple(fr.shape)}", flush=True)

    # leave-one-out NN among the reals: the reference distribution
    sims = fr @ fr.T
    sims.fill_diagonal_(-2.0)
    rr = (1 - sims.max(dim=1).values).cpu().numpy()
    ref = dict(median=float(np.median(rr)), p01=float(np.percentile(rr, 1)),
               p05=float(np.percentile(rr, 5)), mean=float(rr.mean()))
    print(f"real-real LOO NN cosine distance: median {ref['median']:.4f} "
          f"p05 {ref['p05']:.4f} p01 {ref['p01']:.4f}", flush=True)
    del sims
    torch.cuda.empty_cache()

    results = {"real_real_reference": ref, "arms": {}}
    for name, rel in ARMS:
        folder = os.path.join(B, rel)
        fg = torch.nn.functional.normalize(features(folder, extractor, dev), dim=1).to(dev)
        s = fg @ fr.T                              # generated x real
        best = s.max(dim=1)
        d = (1 - best.values).cpu().numpy()
        idx = best.indices.cpu().numpy()
        res = dict(n=len(d), median=float(np.median(d)), mean=float(d.mean()),
                   p01=float(np.percentile(d, 1)), min=float(d.min()),
                   # the memorisation statistic: how much of this arm sits closer to
                   # the training set than a real PR typically sits to its own neighbour
                   frac_below_real_p05=float((d < ref["p05"]).mean()),
                   frac_below_real_median=float((d < ref["median"]).mean()))
        results["arms"][name] = res
        print(f"{name:24s} median {res['median']:.4f}  min {res['min']:.4f}  "
              f"frac<real_p05 {res['frac_below_real_p05']:.4f}  "
              f"frac<real_median {res['frac_below_real_median']:.4f}", flush=True)

        # closest-pair montage
        order = np.argsort(d)[:8]
        gnames = Folder(folder).paths
        tiles = []
        for gi in order:
            g = Image.open(os.path.join(folder, gnames[gi])).convert("L").resize((384, 192))
            r = Image.open(os.path.join(REAL, real_names[idx[gi]])).convert("L").resize((384, 192))
            tiles.append((g, r, d[gi]))
        sheet = Image.new("RGB", (384 * 2 + 12, 192 * len(tiles) + 14 * len(tiles)), "white")
        from PIL import ImageDraw
        dr = ImageDraw.Draw(sheet)
        for k, (g, r, dist) in enumerate(tiles):
            y = k * (192 + 14)
            dr.text((4, y + 2), f"generated (cos dist to nearest real = {dist:.4f})", fill="black")
            dr.text((396, y + 2), "nearest real training image", fill="black")
            sheet.paste(g, (0, y + 14)); sheet.paste(r, (396, y + 14))
        sheet.save(os.path.join(B, "analysis_r25_r31/out", f"nn_{name}.png"))
        del fg, s
        torch.cuda.empty_cache()

    fn = f"memorisation_{_only}.json" if _only else "memorisation.json"
    json.dump(results, open(os.path.join(B, "analysis_r25_r31/out", fn), "w"), indent=2)
    print("wrote", fn, flush=True)


if __name__ == "__main__":
    main()
