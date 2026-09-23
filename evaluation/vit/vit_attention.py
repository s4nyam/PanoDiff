"""
Quantitative attention analysis (reviewer R4 Minor 1b).

The submitted manuscript showed ViT attention maps for a handful of random
samples and argued qualitatively that attention on real radiographs
concentrates on anatomy while attention on synthetic ones is more diffuse. The
reviewer asked for that to be measured over the full test set, within
predefined anatomical regions, with a comparison between the groups.

This script does the whole thing:
  1. trains a compact ViT to separate real (GT-HR) from synthetic (PD-HR)
     radiographs, reproducing the setup described in the paper;
  2. computes attention-rollout maps for every held-out image;
  3. scores each map against the anatomical priors built from the TUFTS
     segmentations -- fraction of attention mass falling on teeth, on the
     maxillomandibular region, and outside it -- plus the map's entropy as a
     scale-free measure of how concentrated the attention is;
  4. compares the two groups with Mann-Whitney U tests and reports effect
     sizes.

The original ViT checkpoint did not survive in the project files, so this is a
newly trained classifier. That is stated in the manuscript: the qualitative
claim is re-tested rather than the original model re-measured.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

T = WORK + "/temp-codes"
BASE = (WORK + "/"
        "project-files/PanoDiff/Syn-Calc-FID&IS")
REAL_DIR = os.path.join(BASE, "10-TrainDataInHR")
SYNTH_DIR = os.path.join(BASE, "4-DiffHATSR")
H, W, P = 256, 512, 16                      # 16 x 32 = 512 patches


# ------------------------------------------------------------------ data ----
class Pairs(Dataset):
    def __init__(self, files, labels):
        self.files, self.labels = files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("L").resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
        return x * 2 - 1, self.labels[i]


def build_split(seed=0, val_frac=0.2):
    ext = (".png", ".jpg", ".jpeg")
    real = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR)
                  if f.lower().endswith(ext))
    syn = sorted(os.path.join(SYNTH_DIR, f) for f in os.listdir(SYNTH_DIR)
                 if f.lower().endswith(ext))
    n = min(len(real), len(syn))
    real, syn = real[:n], syn[:n]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    tr = [real[i] for i in idx[:cut]] + [syn[i] for i in idx[:cut]]
    trl = [0] * cut + [1] * cut
    va = [real[i] for i in idx[cut:]] + [syn[i] for i in idx[cut:]]
    val = [0] * (n - cut) + [1] * (n - cut)
    return (tr, trl), (va, val)


# ------------------------------------------------------------------- ViT ----
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.heads = heads
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x, need_attn=False):
        B, N, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.heads, D // self.heads) \
                             .permute(2, 0, 3, 1, 4)
        attn = None
        if need_attn:
            # explicit softmax only when the map is wanted, so training keeps
            # the fused kernel
            a = (q @ k.transpose(-2, -1)) * (D // self.heads) ** -0.5
            attn = a.softmax(-1)
            o = attn @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(o)
        x = x + self.mlp(self.n2(x))
        return x, attn


class ViT(nn.Module):
    def __init__(self, d=192, depth=6, heads=6):
        super().__init__()
        self.patch = nn.Conv2d(1, d, P, P)
        self.gh, self.gw = H // P, W // P
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.gh * self.gw, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x, need_attn=False):
        B = x.shape[0]
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos
        attns = []
        for b in self.blocks:
            t, a = b(t, need_attn)
            if need_attn:
                attns.append(a)
        return self.head(self.norm(t)[:, 0]), attns


def rollout(attns, gh, gw):
    """Abnar & Zuidema attention rollout: average heads, add the residual
    connection, renormalise, and multiply through the depth."""
    r = None
    for a in attns:
        a = a.mean(1)                                  # average over heads
        a = a + torch.eye(a.shape[-1], device=a.device)  # residual path
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    m = r[:, 0, 1:]                                    # CLS attention to patches
    m = m / m.sum(-1, keepdim=True).clamp_min(1e-12)
    return m.reshape(-1, gh, gw)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=os.path.join(T, "vit_attention.json"))
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (trf, trl), (vaf, val) = build_split()
    print(f"train {len(trf)}  val {len(vaf)}", flush=True)
    dtr = DataLoader(Pairs(trf, trl), batch_size=a.batch, shuffle=True,
                     num_workers=6, pin_memory=True, drop_last=True)
    dva = DataLoader(Pairs(vaf, val), batch_size=a.batch, num_workers=6)

    model = ViT().to(dev)
    print(f"ViT params {sum(p.numel() for p in model.parameters())/1e6:.2f} M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, epochs=a.epochs, steps_per_epoch=len(dtr))

    for ep in range(a.epochs):
        model.train()
        tot = corr = 0
        for x, y in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            corr += (logits.argmax(1) == y).sum().item()
            tot += y.numel()
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in dva:
                x, y = x.to(dev), y.to(dev)
                logits, _ = model(x)
                vc += (logits.argmax(1) == y).sum().item()
                vt += y.numel()
        print(f"epoch {ep+1}/{a.epochs}  train {corr/tot:.4f}  val {vc/vt:.4f}",
              flush=True)
    val_acc = vc / vt

    # ------------------------------------------------- attention statistics --
    pri = np.load(os.path.join(T, "priors", "anatomical_priors.npz"))
    gh, gw = H // P, W // P

    def to_grid(p):
        t = torch.from_numpy(p.astype(np.float32))[None, None]
        return F.adaptive_avg_pool2d(t, (gh, gw))[0, 0].to(dev)

    teeth = (to_grid(pri["teeth"]) > 0.5).float()
    jaw = (to_grid(pri["jaw"]) > 0.5).float()
    bg = 1 - jaw

    recs = []
    model.eval()
    with torch.no_grad():
        for x, y in dva:
            x = x.to(dev)
            logits, attns = model(x, need_attn=True)
            m = rollout(attns, gh, gw)
            prob = logits.softmax(1)[:, 0]           # P(real)
            for i in range(m.shape[0]):
                A = m[i]
                ent = float(-(A * (A + 1e-12).log()).sum())
                recs.append(dict(
                    label=int(y[i]),
                    p_real=float(prob[i]),
                    correct=int((logits[i].argmax() == y[i]).item()),
                    mass_teeth=float((A * teeth).sum()),
                    mass_jaw=float((A * jaw).sum()),
                    mass_bg=float((A * bg).sum()),
                    entropy=ent))

    from scipy import stats as st
    real = [r for r in recs if r["label"] == 0]
    syn = [r for r in recs if r["label"] == 1]
    print(f"\nheld-out set: {len(real)} real, {len(syn)} synthetic, "
          f"classifier accuracy {val_acc:.4f}")
    print("=" * 88)
    print(f"{'measure':16} {'real':>18} {'synthetic':>18} {'p (Mann-Whitney)':>18} {'effect':>8}")
    summary = {}
    for key, label in [("mass_teeth", "attention on teeth"),
                       ("mass_jaw", "attention on jaw"),
                       ("mass_bg", "attention outside jaw"),
                       ("entropy", "attention entropy")]:
        ra = np.array([r[key] for r in real])
        sa = np.array([r[key] for r in syn])
        u, p = st.mannwhitneyu(ra, sa, alternative="two-sided")
        # rank-biserial correlation: a scale-free effect size for Mann-Whitney
        rb = 1 - 2 * u / (len(ra) * len(sa))
        summary[key] = dict(real_mean=float(ra.mean()), real_sd=float(ra.std()),
                            syn_mean=float(sa.mean()), syn_sd=float(sa.std()),
                            p=float(p), rank_biserial=float(rb))
        print(f"{label:16} {ra.mean():8.4f}+-{ra.std():.4f} "
              f"{sa.mean():8.4f}+-{sa.std():.4f} {p:18.3e} {rb:+8.3f}")

    # uniform-attention reference: what the region masses would be if attention
    # were spread evenly, i.e. carrying no anatomical preference at all
    unif = {"mass_teeth": float(teeth.mean()), "mass_jaw": float(jaw.mean()),
            "mass_bg": float(bg.mean()),
            "entropy": float(np.log(gh * gw))}
    print(f"\nuniform-attention reference: teeth {unif['mass_teeth']:.4f}, "
          f"jaw {unif['mass_jaw']:.4f}, outside {unif['mass_bg']:.4f}, "
          f"entropy {unif['entropy']:.4f}")

    json.dump({"val_accuracy": val_acc, "n_real": len(real), "n_syn": len(syn),
               "summary": summary, "uniform_reference": unif,
               "config": {"H": H, "W": W, "patch": P, "epochs": a.epochs}},
              open(a.out, "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(T, "vit_realfake.pt"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
