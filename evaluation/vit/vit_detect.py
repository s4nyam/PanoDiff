"""Per-arm machine detectability, using the paper's OWN ViT protocol.

The manuscript already trains a compact ViT to separate real PRs from PanoDiff-SR
PRs and reports 97.7% test accuracy. Running that identical protocol against every
baseline turns "which images look real" into a measured quantity on the same axis
the paper already uses, and it is the axis on which FID and realism can disagree:
a model can match the reference distribution well (low FID) while still leaving
per-image artefacts a classifier picks up immediately.

One arm per SLURM task. Reuses the ViT, the 256x512 / patch-16 geometry and the
attention-rollout code from temp-codes/vit_attention.py rather than reimplementing
them, so the numbers are comparable with the figure already in the paper.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

T = WORK + "/temp-codes"
sys.path.insert(0, T)
from vit_attention import ViT, rollout, H, W, P          # noqa: E402

B = WORK + "/new-baselines-generation"
REAL = os.path.join(B, "refs/real")

ARMS = [
    ("PanoDiffSR_existing",    "refs/PanoDiffSR_existing"),
    ("sg2ada",                 "samples/sg2ada"),
    ("adm",                    "samples/adm"),
    ("ldm",                    "samples/ldm"),
    ("panodiff_hr",            "samples/panodiff_hr"),
    ("FastGAN_HATSR_existing", "refs/FastGAN_HATSR_existing"),
    ("fastgan_lite64",         "samples/fastgan_lite64"),
    ("sg3",                    "samples/sg3"),
]
# ONLY_ARM re-runs a single arm without disturbing the numbers already reported
# for the others (the classifier is seeded, but re-running all six needlessly
# risks a diff nobody asked for).
_only = os.environ.get("ONLY_ARM")
if _only:
    ARMS = [a for a in ARMS if a[0] == _only]
EPOCHS, BATCH, LR = 8, 32, 3e-4


class Pairs(Dataset):
    def __init__(self, files, labels):
        self.files, self.labels = files, labels
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("L").resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
        return x * 2 - 1, self.labels[i]


def build_split(synth_dir, seed=0, val_frac=0.2):
    """Same 80:20 split rule as the paper's script, applied to this arm."""
    ext = (".png", ".jpg", ".jpeg")
    real = sorted(os.path.join(REAL, f) for f in os.listdir(REAL) if f.lower().endswith(ext))
    syn = sorted(os.path.join(synth_dir, f) for f in os.listdir(synth_dir) if f.lower().endswith(ext))
    n = min(len(real), len(syn))
    real, syn = real[:n], syn[:n]
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - val_frac))
    tr = [real[i] for i in idx[:cut]] + [syn[i] for i in idx[:cut]]
    va = [real[i] for i in idx[cut:]] + [syn[i] for i in idx[cut:]]
    return (tr, [0]*cut + [1]*cut), (va, [0]*(n-cut) + [1]*(n-cut))


def auc(scores, labels):
    """Rank-based AUC; scores = P(synthetic), labels 1 = synthetic."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    npos, nneg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local = int(os.environ.get("SLURM_LOCALID", 0))
    if rank >= len(ARMS):
        print(f"rank {rank}: no arm", flush=True); return
    name, rel = ARMS[rank]
    synth = os.path.join(B, rel)
    dev = torch.device(f"cuda:{local}")
    torch.manual_seed(0)

    (trf, trl), (vaf, val) = build_split(synth)
    dtr = DataLoader(Pairs(trf, trl), batch_size=BATCH, shuffle=True,
                     num_workers=6, pin_memory=True, drop_last=True)
    dva = DataLoader(Pairs(vaf, val), batch_size=BATCH, num_workers=6)
    print(f"[{name}] train {len(trf)} val {len(vaf)}", flush=True)

    model = ViT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS,
                                                steps_per_epoch=len(dtr))
    for ep in range(EPOCHS):
        model.train()
        for x, y in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = F.cross_entropy(model(x)[0], y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        model.eval(); vc = vt = 0
        with torch.no_grad():
            for x, y in dva:
                x, y = x.to(dev), y.to(dev)
                vc += (model(x)[0].argmax(1) == y).sum().item(); vt += y.numel()
        print(f"[{name}] epoch {ep+1}/{EPOCHS} val {vc/vt:.4f}", flush=True)

    # final pass: accuracy, AUC and the anatomical attention split
    pri = np.load(os.path.join(T, "priors", "anatomical_priors.npz"))
    gh, gw = H // P, W // P
    def to_grid(p):
        t = torch.from_numpy(p.astype(np.float32))[None, None]
        return F.adaptive_avg_pool2d(t, (gh, gw))[0, 0].to(dev)
    teeth = (to_grid(pri["teeth"]) > 0.5).float()
    jaw = (to_grid(pri["jaw"]) > 0.5).float()

    sc, lb, mt_r, mt_s, ent_r, ent_s = [], [], [], [], [], []
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in dva:
            x = x.to(dev)
            logits, attns = model(x, need_attn=True)
            m = rollout(attns, gh, gw)
            p_syn = logits.softmax(1)[:, 1]
            correct += (logits.argmax(1).cpu() == y).sum().item(); total += y.numel()
            for i in range(x.shape[0]):
                A = m[i]
                sc.append(float(p_syn[i])); lb.append(int(y[i]))
                e = float(-(A * (A + 1e-12).log()).sum())
                (mt_s if y[i] == 1 else mt_r).append(float((A * teeth).sum()))
                (ent_s if y[i] == 1 else ent_r).append(e)

    sc, lb = np.array(sc), np.array(lb)
    res = dict(arm=name, n_train=len(trf), n_val=len(vaf),
               accuracy=correct / total, auc=float(auc(sc, lb)),
               attn_teeth_real=float(np.mean(mt_r)), attn_teeth_syn=float(np.mean(mt_s)),
               attn_entropy_real=float(np.mean(ent_r)), attn_entropy_syn=float(np.mean(ent_s)))
    print(f"[{name}] ACC {res['accuracy']:.4f}  AUC {res['auc']:.4f}", flush=True)
    json.dump(res, open(os.path.join(B, "analysis_r25_r31/out", f"vit_{name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
