"""The two qualitative mosaics of Section 4.5 (comparison with alternative models), upright.

Layout rule for both: METHODS are rows, SAMPLES are columns, three columns across the
full text width (390 pt), so each 2:1 radiograph is 130 x 65 pt -- about 40% larger
than the original layouts. Each row's label sits in a thin horizontal strip ABOVE the
row rather than in a side column, which is what frees the width. Nothing is rotated:
an earlier 90-degree version was the same size but unreadable in practice.

  lr_sr_mosaic.pdf    replaces the old Figures 14 and 15. One seed per column, traced
                      from the low-resolution sample through HAT-SR and SwinIR, for
                      PanoDiff and FastGAN. A zoom inset marks the same region in
                      every panel.
  directhr_mosaic.pdf the direct-high-resolution comparison, three images per row drawn at
                      random in the same way for every row, the real radiographs and
                      PanoDiff-SR included: the picks come from select_random_examples.py
                      (random_examples.json), which takes the first three images of the
                      random 1500-image sample behind the anatomical table. No image is
                      chosen or replaced by eye (since 2026-10-13; before that the rows
                      showed selected examples).

Run inside the LUMI PyTorch container (needs PIL + matplotlib).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

LEG = WORK + "/data/original-corpus/PanoDiff/Syn-Calc-FID&IS"
NB = WORK + "/experiments/direct-hr-baselines"
# SwinIR fine-tuned on the 7243 PR pairs (super_resolution/swinir/infer_swinir_ft.py), as used in Section 4.5.1
SWFT = WORK + "/experiments/sr-distribution-metrics/out"
FIGS = WORK + "/manuscript/overleaf/figures"

PT = 1 / 72.27
W_PT = 390.0          # \textwidth of the elsarticle preprint at 12pt
HEADER_PT = 10.0      # label strip above each row
NCOL = 3
PANEL_W_PT = W_PT / NCOL
PANEL_H_PT = PANEL_W_PT / 2.0                    # PRs are 1024x512

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def load_display(path):
    """Grayscale at 1024x512; LR inputs are upsampled bicubically for display."""
    im = Image.open(path).convert("L")
    if im.size != (1024, 512):
        im = im.resize((1024, 512), Image.BICUBIC)
    return np.asarray(im)


def frame(ax, arr):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.6")


def figure_for(n_rows):
    h_pt = n_rows * (HEADER_PT + PANEL_H_PT)
    return plt.figure(figsize=(W_PT * PT, h_pt * PT)), h_pt


def panel_box(r, c, h_pt):
    """Axes rectangle (figure fraction) for row r, column c, below that row's header."""
    top = h_pt - r * (HEADER_PT + PANEL_H_PT) - HEADER_PT
    return [c / NCOL, (top - PANEL_H_PT) / h_pt, 1 / NCOL, PANEL_H_PT / h_pt]


def row_header(fig, r, h_pt, text, bold=False):
    y = (h_pt - r * (HEADER_PT + PANEL_H_PT) - HEADER_PT / 2) / h_pt
    fig.text(0.002, y, text, ha="left", va="center", fontsize=7.2,
             fontweight="bold" if bold else "normal")


# ------------------------------------------------ LR synthesis + super-resolution
ROI = (700, 200, 800, 300)   # x0, y0, x1, y1 in 1024x512 -- the region the old insets used


def lr_sr_mosaic():
    diff = ["001417", "001442", "001458"]
    gan = ["0000166", "0000547", "0000586"]
    rows = [
        ("PanoDiff: low-resolution sample ($256\\times128$, shown upsampled)",
         lambda i: f"{LEG}/2-DiffLR/generated_{diff[i]}.png"),
        ("PanoDiff + HAT-SR ($1024\\times512$)",
         lambda i: f"{LEG}/4-DiffHATSR/generated_{diff[i]}__SR.png"),
        ("PanoDiff + SwinIR ($1024\\times512$)",
         lambda i: f"{SWFT}/6ft-DiffSwinIRft/generated_{diff[i]}_SwinIR.png"),
        ("FastGAN: low-resolution sample ($256\\times128$, shown upsampled)",
         lambda i: f"{LEG}/3-GANsLR/{gan[i]}.png"),
        ("FastGAN + HAT-SR ($1024\\times512$)",
         lambda i: f"{LEG}/5-GANsHATSR/{gan[i]}__SR.png"),
        ("FastGAN + SwinIR ($1024\\times512$)",
         lambda i: f"{SWFT}/7ft-GANsSwinIRft/{gan[i]}_SwinIR.png"),
    ]
    fig, h_pt = figure_for(len(rows))
    x0, y0, x1, y1 = ROI
    for r, (name, path) in enumerate(rows):
        row_header(fig, r, h_pt, name)
        for c in range(NCOL):
            ax = fig.add_axes(panel_box(r, c, h_pt))
            arr = load_display(path(c))
            frame(ax, arr)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="#ffd21f", lw=0.6))
            # Square inset in page units: 60% of the panel height, top-left corner.
            ins = ax.inset_axes([0.015, 0.37, 0.30, 0.60])
            ins.imshow(arr[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                       interpolation="nearest", aspect="auto")
            ins.set_xticks([]); ins.set_yticks([])
            for s in ins.spines.values():
                s.set_linewidth(0.7); s.set_color("#ffd21f")
        if r == 3:   # separate the two generator families
            yy = (h_pt - 3 * (HEADER_PT + PANEL_H_PT)) / h_pt
            fig.add_artist(plt.Line2D([0, 1], [yy, yy], color="0.2", lw=0.9))
    out = f"{FIGS}/fig15-lr-sr-mosaic"
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/lr_sr_mosaic.pdf", dpi=400, bbox_inches=None)
    fig.savefig(f"{out}/lr_sr_mosaic.png", dpi=170)
    plt.close(fig)
    print("wrote", f"{out}/lr_sr_mosaic.pdf")


# ------------------------------------------------------ direct high-resolution
def directhr_mosaic(out=None):
    rows = [
        ("Real radiographs",                                       "refs/real"),
        ("PanoDiff-SR (two-stage, this work)",                    "refs/PanoDiffSR_existing"),
        ("StyleGAN2-ADA (direct, $1024\\times512$)",               "samples/sg2ada"),
        ("StyleGAN3 (direct, $1024\\times512$)",                   "samples/sg3"),
        ("ADM (direct, $1024\\times512$)",                         "samples/adm"),
        ("LDM (direct, $1024\\times512$)",                         "samples/ldm"),
        ("PanoDiff-HR (direct, $1024\\times512$)",                 "samples/panodiff_hr"),
    ]
    # Three random images per row, the same draw for every row (select_random_examples.py).
    picks = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_examples.json")))["picks"]
    key = {"refs/real": "real", "refs/PanoDiffSR_existing": "PanoDiffSR_existing", "samples/sg2ada": "sg2ada",
           "samples/sg3": "sg3", "samples/adm": "adm", "samples/ldm": "ldm", "samples/panodiff_hr": "panodiff_hr"}
    fig, h_pt = figure_for(len(rows))
    for r, (name, rel) in enumerate(rows):
        row_header(fig, r, h_pt, name, bold=(r == 1))
        pick = picks[key[rel]]
        assert len(pick) == NCOL and all(os.path.dirname(p) == os.path.join(NB, rel) for p in pick), (rel, pick)
        for c, p in enumerate(pick):
            ax = fig.add_axes(panel_box(r, c, h_pt))
            frame(ax, load_display(p))
    out = out or f"{FIGS}/fig17-directhr"
    fig.savefig(f"{out}/directhr_mosaic.pdf", dpi=400, bbox_inches=None)
    fig.savefig(f"{out}/directhr_mosaic.png", dpi=170)
    plt.close(fig)
    print("wrote", f"{out}/directhr_mosaic.pdf")


if __name__ == "__main__":
    import sys as _s
    # --directhr-only: leave the LR/SR mosaic as it is; --out DIR: write the direct-HR mosaic to DIR
    if "--directhr-only" not in _s.argv:
        lr_sr_mosaic()
    directhr_mosaic(_s.argv[_s.argv.index("--out") + 1] if "--out" in _s.argv else None)
