"""Figure 15 (MedSAM probing subsection): real and synthetic radiographs, twelve of each (three
rows of four per set), each prompted with the dentition box, i.e. the box around the whole U-Net
tooth region, exactly as the probe's "arch" protocol builds it. The yellow box is the prompt, the
cyan contour is MedSAM's mask, and the badge is its predicted mask quality (the same value the
probe records for that image).

The 24 radiographs are listed in medsam_figure_picks.json (indices into the arrays written by
prepare_data_crop.py; see its "criterion"): complete dentitions were shortlisted automatically by
one rule applied to both sets, and the twelve of each set with a full or nearly full dentition on
a clean radiograph were then chosen by hand, three per device on the real rows.

Until 6 Oct 2026 this figure showed one box per connected tooth region of the U-Net mask; that
prompt is no longer reported (a region can hold several teeth and is not an anatomical unit).

Layout: four columns across the full text width (390 pt), each panel a 2:1 radiograph
97.5 x 48.75 pt, a 9 pt label strip above the first row of each set. CPU is fine (24 image
encodings, about two minutes).
Environment: PROBE_DATA / PROBE_REF (default medsam-probe/data_crop), PROBE_OUT (where the badge
values are written, default medsam-probe/out_crop), FIG_OUT (default the manuscript's figure
folder), PICKS (default medsam_figure_picks.json next to this script).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medsam_probe import load_medsam, embed, prompt, H, W, E   # noqa: E402

PDATA = os.environ.get("PROBE_DATA", f"{E}/medsam-probe/data_crop")
PREF = os.environ.get("PROBE_REF", f"{E}/medsam-probe/data_crop")
POUT = os.environ.get("PROBE_OUT", f"{E}/medsam-probe/out_crop")
FIGS = os.environ.get("FIG_OUT", f"{E}/new-files/overleaf/figures/fig20-medsam")
PICKS = os.environ.get("PICKS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "medsam_figure_picks.json"))
PT = 1 / 72.27
W_PT, NCOL, NROW, HEADER_PT = 390.0, 4, 3, 9.0
PANEL_W, PANEL_H = W_PT / NCOL, W_PT / NCOL / 2

d0 = json.load(open(PICKS))
real = [("real", r["idx"], r["source"]) for r in d0["real"]]
syn = [("syn", r["idx"], "PanoDiff-SR") for r in d0["syn"]]
assert len(real) == NROW * NCOL and len(syn) == NROW * NCOL, (len(real), len(syn))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = load_medsam(dev)
imgs = {"real": np.load(f"{PDATA}/real_img.npy", mmap_mode="r"), "syn": np.load(f"{PDATA}/syn_img.npy", mmap_mode="r")}
refs = {"real": np.load(f"{PREF}/ref_real.npy", mmap_mode="r"),
        "syn": np.load(f"{PREF}/ref_syn.npy", mmap_mode="r")}

ROWS = [("Real radiographs (three per device)", real[:NCOL])] + [(None, real[r * NCOL:(r + 1) * NCOL]) for r in range(1, NROW)] \
     + [("PanoDiff-SR", syn[:NCOL])] + [(None, syn[r * NCOL:(r + 1) * NCOL]) for r in range(1, NROW)]
strip = [HEADER_PT if lab else 2.0 for lab, _ in ROWS]
h_pt = sum(strip) + len(ROWS) * PANEL_H
fig = plt.figure(figsize=(W_PT * PT, h_pt * PT))
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
scores = []
for r, (label, picks) in enumerate(ROWS):
    top = h_pt - sum(strip[:r + 1]) - r * PANEL_H
    if label:
        fig.text(0.002, (top + HEADER_PT / 2) / h_pt, label, ha="left", va="center", fontsize=7.2)
    for c, (which, i, src) in enumerate(picks):
        img = np.asarray(imgs[which][i]); ref = np.asarray(refs[which][i]) > 0
        ys, xs = np.nonzero(ref)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]          # the probe's "arch" box
        masks, ious = prompt(sam, embed(sam, img, dev), [box], dev)
        scores.append({"set": which, "idx": int(i), "source": src, "pred_iou": float(ious[0])})
        ax = fig.add_axes([c / NCOL, (top - PANEL_H) / h_pt, 1 / NCOL, PANEL_H / h_pt])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", aspect="auto")
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec="#ffd21f", lw=0.45))
        ax.contour(masks[0].astype(float), levels=[0.5], colors=["#19d3ff"], linewidths=0.5)
        ax.text(0.985, 0.05, f"{float(ious[0]):.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        if which == "real":
            ax.text(0.015, 0.05, src, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.6")
os.makedirs(FIGS, exist_ok=True)
fig.savefig(f"{FIGS}/medsam_probe.pdf", dpi=600)
fig.savefig(f"{FIGS}/medsam_probe.png", dpi=170)
os.makedirs(POUT, exist_ok=True)
json.dump(scores, open(f"{POUT}/medsam_figure_badges.json", "w"), indent=1)
print("wrote", f"{FIGS}/medsam_probe.pdf")
