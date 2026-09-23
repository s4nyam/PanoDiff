"""Process figure for the attention statistics of Section 4.3.

Top row, left to right: a held-out radiograph; the attention-rollout map of the real-versus-
synthetic ViT for it; the fixed anatomical templates (the 1000 TUFTS teeth and
maxillomandibular masks averaged and thresholded at 0.5) with the three numbers that follow
for this image. Bottom row: the distributions of those numbers over the 1449 real and 1449
synthetic held-out radiographs, recomputed here with the saved classifier
(temp-codes/vit_realfake.pt) so the figure and the text come from the same model.

Runs on CPU inside the LUMI PyTorch container: the ViT has 2.8 M parameters.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from torch.utils.data import DataLoader

T = WORK + "/temp-codes"
OUT = WORK + "/new-files/overleaf/figures/fig21-attn-process"
sys.path.insert(0, T)
from vit_attention import ViT, Pairs, build_split, rollout, H, W, P   # noqa: E402

PT = 1 / 72.27
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 6.5})
gh, gw = H // P, W // P

model = ViT().eval()
model.load_state_dict(torch.load(os.path.join(T, "vit_realfake.pt"), map_location="cpu"))
pri = np.load(os.path.join(T, "priors", "anatomical_priors.npz"))
def to_grid(p):
    return F.adaptive_avg_pool2d(torch.from_numpy(p.astype(np.float32))[None, None], (gh, gw))[0, 0]
teeth_g = (to_grid(pri["teeth"]) > 0.5).float(); jaw_g = (to_grid(pri["jaw"]) > 0.5).float()

_, (vaf, val) = build_split()
recs, example = [], None
with torch.no_grad():
    for x, y in DataLoader(Pairs(vaf, val), batch_size=64, num_workers=4):
        logits, attns = model(x, need_attn=True)
        m = rollout(attns, gh, gw)
        for i in range(len(x)):
            A = m[i]
            recs.append((int(y[i]), float((A * teeth_g).sum()), float((A * jaw_g).sum()),
                         float(-(A * (A + 1e-12).log()).sum())))
            if example is None and int(y[i]) == 0:
                example = (x[i, 0].numpy(), A.numpy(), recs[-1])
R = np.array(recs)
real, syn = R[R[:, 0] == 0], R[R[:, 0] == 1]
summary = {k: {"real": float(real[:, j].mean()), "syn": float(syn[:, j].mean())}
           for j, k in ((1, "teeth"), (2, "jaw"), (3, "entropy"))}
print("recomputed:", json.dumps(summary), "n", len(real), len(syn))

# ------------------------------------------------------------------ layout
W_PT, PW, PH, GAP = 390.0, 112.0, 56.0, 27.0          # three 2:1 panels with arrow gaps
TOP_Y, BOT_H, H_PT = 128.0, 66.0, 204.0
fig = plt.figure(figsize=(W_PT * PT, H_PT * PT))
def ax_at(x, y, w, h):
    return fig.add_axes([x / W_PT, y / H_PT, w / W_PT, h / H_PT])
img, A, (_, mt, mj, ent) = example
img8 = (img + 1) / 2
xs = [0.0, PW + GAP, 2 * (PW + GAP)]
titles = ["1  Held-out radiograph", "2  Attention rollout (ViT)", "3  Fixed templates"]
for k, x0 in enumerate(xs):
    ax = ax_at(x0, TOP_Y, PW, PH)
    ax.imshow(img8, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    if k == 1:
        up = F.interpolate(torch.from_numpy(A)[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(up / up.max(), cmap="magma", alpha=0.55, aspect="auto", interpolation="bilinear")
    if k == 2:
        yy, xx = np.mgrid[0:H, 0:W]
        tp = F.interpolate(torch.from_numpy(pri["teeth"].astype(np.float32))[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
        jp = F.interpolate(torch.from_numpy(pri["jaw"].astype(np.float32))[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
        ax.contourf(xx, yy, jp > 0.5, levels=[0.5, 1.5], colors=["#4a90d9"], alpha=0.30)
        ax.contourf(xx, yy, tp > 0.5, levels=[0.5, 1.5], colors=["#ffd21f"], alpha=0.45)
        ax.contour(xx, yy, jp, levels=[0.5], colors=["#4a90d9"], linewidths=0.7)
        ax.contour(xx, yy, tp, levels=[0.5], colors=["#ffd21f"], linewidths=0.7, linestyles="dashed")
        fig.text(0.5, (TOP_Y - 9) / H_PT, f"for this image: share of attention on teeth {mt:.3f}, on jaw {mj:.3f}; "
                 f"entropy of the map {ent:.2f} nats", ha="center", va="top", fontsize=6.3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.5")
    fig.text((x0 + PW / 2) / W_PT, (TOP_Y + PH + 4) / H_PT, titles[k], ha="center", va="bottom", fontsize=6.8, fontweight="bold")
for x0 in xs[:2]:
    fig.add_artist(FancyArrowPatch((( x0 + PW + 4) / W_PT, (TOP_Y + PH / 2) / H_PT), ((x0 + PW + GAP - 4) / W_PT, (TOP_Y + PH / 2) / H_PT),
                                   arrowstyle="-|>", mutation_scale=7, lw=0.8, color="0.25", transform=fig.transFigure))
fig.text((xs[0] + PW + GAP / 2) / W_PT, (TOP_Y + PH / 2 + 5) / H_PT, "ViT", ha="center", va="bottom", fontsize=6.0, color="0.25")
fig.text((xs[1] + PW + GAP / 2) / W_PT, (TOP_Y + PH / 2 + 5) / H_PT, "score", ha="center", va="bottom", fontsize=6.0, color="0.25")

# bottom: distributions over the held-out set
fig.text(0.0, (BOT_H + 16) / H_PT, "4  Over 1449 real and 1449 synthetic held-out radiographs", ha="left", va="bottom", fontsize=6.8, fontweight="bold")
BW = (W_PT - 2 * 14) / 3
for k, (j, name, xl) in enumerate(((3, "entropy of the map (nats)", None), (1, "share of attention on teeth", None), (2, "share of attention on jaw", None))):
    ax = ax_at(k * (BW + 14), 22.0, BW, BOT_H - 10)
    lo, hi = np.percentile(R[:, j], [0.5, 99.5]); bins = np.linspace(lo, hi, 36)
    ax.hist(real[:, j], bins=bins, color="0.35", alpha=0.75, label="real", lw=0)
    ax.hist(syn[:, j], bins=bins, color="#1f5fa8", alpha=0.6, label="synthetic", lw=0)
    ax.set_xlabel(name, fontsize=6.3); ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=5.8, length=2)
    if k == 0:
        ax.legend(frameon=False, fontsize=6.0, loc="upper left")
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/attention_process.pdf", dpi=600)
fig.savefig(f"{OUT}/attention_process.png", dpi=180)
json.dump(summary, open(f"{OUT}/recomputed_summary.json", "w"), indent=1)
print("wrote", f"{OUT}/attention_process.pdf")
