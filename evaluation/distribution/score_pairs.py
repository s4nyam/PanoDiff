"""FID / IS for the fine-tuned SwinIR arms, with the paper's own protocol (fid.py / is.py):
torchmetrics FrechetInceptionDistance(normalize=True) and InceptionScore(normalize=True),
ToTensor() only (torchmetrics resizes to 299 internally), batch 64, every image of each folder.

One task per SLURM rank. Two tasks re-score existing paper values as a protocol check:
FID(10-TrainDataInHR, 6-DiffSwinIR) = 91.10 and IS(6-DiffSwinIR) = 1.557 in the paper.
Each task writes out/score_parts/<label>.json.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os, sys
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from PIL import Image

P = WORK
LEG = f"{P}/project-files/PanoDiff/Syn-Calc-FID&IS"
NEW = f"{P}/swinir-ft-eval/out"
F = {  # folder aliases
    "10": f"{LEG}/10-TrainDataInHR", "4": f"{LEG}/4-DiffHATSR", "5": f"{LEG}/5-GANsHATSR",
    "6": f"{LEG}/6-DiffSwinIR",
    "6ft": f"{NEW}/6ft-DiffSwinIRft", "7ft": f"{NEW}/7ft-GANsSwinIRft", "9ft": f"{NEW}/9ft-TrainSwinIRft",
}
TASKS = [
    ("fid", "10", "6ft"), ("fid", "10", "7ft"),
    ("fid", "4", "6ft"), ("fid", "4", "7ft"),
    ("fid", "5", "6ft"), ("fid", "5", "7ft"),
    ("fid", "6ft", "7ft"),
    ("is", "6ft", None), ("is", "7ft", None), ("is", "9ft", None),
    ("fid", "10", "6"),          # protocol check: paper 91.10
    ("is", "6", None),           # protocol check: paper 1.557
]


class Folder(Dataset):
    def __init__(self, d):
        self.d = d
        self.p = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.t = T.ToTensor()
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.t(Image.open(os.path.join(self.d, self.p[i])).convert("RGB"))


def loader(d):
    return DataLoader(Folder(d), batch_size=64, shuffle=False, num_workers=6, pin_memory=True)


@torch.no_grad()
def main():
    rank = int(os.environ.get("SLURM_PROCID", 0)); local = int(os.environ.get("SLURM_LOCALID", 0))
    if rank >= len(TASKS):
        return
    kind, a, b = TASKS[rank]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # ROCR_VISIBLE_DEVICES leaves one GCD
    torch.manual_seed(0)
    label = f"{kind}_{a}" + (f"_vs_{b}" if b else "")
    if kind == "fid":
        m = FrechetInceptionDistance(normalize=True).to(dev)
        for x in loader(F[a]): m.update(x.to(dev), real=True)
        for x in loader(F[b]): m.update(x.to(dev), real=False)
        res = {"fid": float(m.compute()), "n_a": len(Folder(F[a])), "n_b": len(Folder(F[b]))}
    else:
        m = InceptionScore(normalize=True).to(dev)
        for x in loader(F[a]): m.update(x.to(dev))
        mean, std = m.compute()
        res = {"is_mean": float(mean), "is_std": float(std), "n": len(Folder(F[a]))}
    res.update(task=label)
    os.makedirs(f"{NEW}/score_parts", exist_ok=True)
    json.dump(res, open(f"{NEW}/score_parts/{label}.json", "w"), indent=1)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
