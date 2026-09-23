"""Score every baseline exactly as the paper's own numbers were scored.

Comparability is the whole point of this file, so it deliberately reuses the
protocol of project-files/PanoDiff/Syn-Calc-FID&IS/{fid.py,is.py} rather than a
"better" one:
  * torchmetrics FrechetInceptionDistance(normalize=True) / InceptionScore(normalize=True)
  * images read from a folder as PNG, ToTensor() only -- NO manual resize
    (torchmetrics does its own 299x299 Inception resize internally)
  * batch size 64, the full folder on both sides
  * the real reference is always 10-TrainDataInHR, the same 7243 real 1024x512 PRs

It also re-scores the paper's OWN arms (4-DiffHATSR = PanoDiff-SR, 5-GANsHATSR =
FastGAN+HAT-SR) in the same pass. The published Table 3 says HRGT-HRPD = 40.7
while supplementary Table 8 says 40.51 for what should be the same pair, so
re-deriving every row here means the new comparison table is internally
consistent and does not inherit that discrepancy.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, csv, os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image

REAL = (WORK + "/new-baselines-generation/refs/real")


class Folder(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.paths = sorted(f for f in os.listdir(folder)
                            if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = T.Compose([T.ToTensor()])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.t(Image.open(os.path.join(self.folder, self.paths[i])).convert("RGB"))


_REAL_CACHE = {}


def _real_features(dev, batch):
    """The real side is identical for every arm, so its 7243 Inception passes are
    computed once and the accumulated statistics reused. Scoring 7 folders
    otherwise repeats that work 7 times."""
    if "f" not in _REAL_CACHE:
        fid = FrechetInceptionDistance(normalize=True).to(dev)
        with torch.no_grad():
            for b in DataLoader(Folder(REAL), batch_size=batch, num_workers=6, pin_memory=True):
                fid.update(b.to(dev), real=True)
        _REAL_CACHE["f"] = (fid.real_features_sum.clone(),
                            fid.real_features_cov_sum.clone(),
                            fid.real_features_num_samples.clone())
    return _REAL_CACHE["f"]


def score(folder, device, batch=64):
    dev = torch.device(device)
    fid = FrechetInceptionDistance(normalize=True).to(dev)
    rs, rc, rn = _real_features(dev, batch)
    fid.real_features_sum, fid.real_features_cov_sum, fid.real_features_num_samples = \
        rs.clone(), rc.clone(), rn.clone()
    iscore = InceptionScore(normalize=True).to(dev)
    with torch.no_grad():
        for b in DataLoader(Folder(folder), batch_size=batch, num_workers=6, pin_memory=True):
            b = b.to(dev)
            fid.update(b, real=False)
            iscore.update(b)
    m, s = iscore.compute()
    return fid.compute().item(), m.item(), s.item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", required=True,
                    help="name=path pairs, e.g. panodiff_hr=/path/to/samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rows = []
    for spec in a.folders:
        name, path = spec.split("=", 1)
        n = len([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        f, im, isd = score(path, a.device)
        print(f"{name:22s} n={n:5d}  FID {f:8.2f}  IS {im:.3f} +/- {isd:.3f}", flush=True)
        rows.append(dict(model=name, n_images=n, fid=round(f, 4),
                         is_mean=round(im, 4), is_std=round(isd, 4)))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "n_images", "fid", "is_mean", "is_std"])
        w.writeheader(); w.writerows(rows)
    print("wrote", a.out)
