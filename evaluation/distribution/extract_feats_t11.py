"""[Table 11 arms] One Inception pass per image set -> cached features for FID, KID, IS and precision/recall.

Uses torchmetrics' own network and input handling so the numbers are those of the paper's
fid.py / is.py: NoTrainInceptionV3('inception-v3-compat'), images read with ToTensor() at native
size, normalize=True semantics (x*255 -> uint8), internal resize to 299. Both feature layers are
taken in the same pass: '2048' (pool, used by FID/KID/PR) and 'logits_unbiased' (used by IS).

One set per SLURM rank; writes out/feats/<set>.npz (pool float32 [N,2048], logits float32 [N,1008]).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os, sys
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchmetrics.image.fid import NoTrainInceptionV3

P = WORK
LEG = f"{P}/project-files/PanoDiff/Syn-Calc-FID&IS"
NEW = f"{P}/swinir-ft-eval/out"
# (name, folder, max images). Row 5 is cut to the 7243 FastGAN seeds that rows 3 and 7 use.
# Table 11 direct high-resolution arms (the two-stage rows 4 and 5 and the real set 10 are
# already in out/feats from extract_feats.py).
NB = WORK + "/new-baselines-generation/samples"
SETS = [("sg2ada", f"{NB}/sg2ada", None), ("sg3", f"{NB}/sg3", None),
        ("fastgan_lite64", f"{NB}/fastgan_lite64", None), ("adm", f"{NB}/adm", None),
        ("ldm", f"{NB}/ldm", None), ("panodiff_hr", f"{NB}/panodiff_hr", None)]


class Folder(Dataset):
    def __init__(self, d, cap):
        p = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.d, self.p, self.t = d, (p[:cap] if cap else p), T.ToTensor()
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.t(Image.open(os.path.join(self.d, self.p[i])).convert("RGB"))


@torch.no_grad()
def main():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    if rank >= len(SETS):
        return
    name, folder, cap = SETS[rank]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048", "logits_unbiased"]).to(dev).eval()
    ds = Folder(folder, cap)
    pools, logits = [], []
    for x in DataLoader(ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True):
        x = (x.to(dev) * 255).byte()                       # torchmetrics normalize=True
        f, l = (t.reshape(x.shape[0], -1) for t in net._torch_fidelity_forward(x))   # forward() returns only the first layer
        pools.append(f.float().cpu()); logits.append(l.float().cpu())
    os.makedirs(f"{NEW}/feats", exist_ok=True)
    np.savez(f"{NEW}/feats/{name}.npz", pool=torch.cat(pools).numpy(), logits=torch.cat(logits).numpy(),
             files=np.array(ds.p))
    print(f"{name}: {len(ds)} images from {folder}", flush=True)


if __name__ == "__main__":
    main()
