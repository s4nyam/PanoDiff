"""Figure-17-style precision (fidelity) / recall (coverage) curves for every model of Table 11.

Reads swinir-ft-eval/out/pr_ratios_t11.npz (code/pr_ratios_t11.py). Same construction as
make_pr_scores.py: per-image distance to the nearest image of the other set relative to its
k = 3 neighbourhood radius; the share left of the dashed line is the precision or recall.
Column (a): the GAN arms, column (b): the diffusion arms; PanoDiff-SR is drawn in both.
NOT a manuscript figure -- written to figure-sources/pr_scores_t11/ for inspection.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SRC = WORK + "/swinir-ft-eval/out"
OUT = WORK + "/new-files/figure-sources/pr_scores_t11/pr_scores_t11"

PD = ("4", "PanoDiff-SR (1)", "#d1495b")
COLS = [
    ("(a) GANs, with PanoDiff-SR for reference",
     [PD, ("5", "FastGAN + HAT-SR (2)", "#E69F00"), ("sg2ada", "StyleGAN2-ADA (3)", "#0072B2"),
      ("sg3", "StyleGAN3 (4)", "#009E73"), ("fastgan_lite64", "FastGAN-HR (5)", "#999999")]),
    ("(b) Diffusion models",
     [PD, ("adm", "ADM (6)", "#CC79A7"), ("ldm", "LDM (7)", "#8c564b"),
      ("panodiff_hr", "PanoDiff-HR (8)", "#56B4E9")]),
]
ROWS = [("fid", "Fidelity (precision)", "each generated image"),
        ("cov", "Coverage (recall)", "each real image")]

PT = 1 / 72.27
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.8, "axes.labelsize": 7, "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

d = np.load(f"{SRC}/pr_ratios_t11.npz")
allv = np.concatenate([d[f"{k}_{t}"] for _, sets in COLS for k, _, _ in sets for t, _, _ in ROWS])
XLIM = (0.8, 1.52)
fig, axes = plt.subplots(2, 2, figsize=(390 * PT, 290 * PT), sharex=True)
fig.subplots_adjust(left=0.075, right=0.995, top=0.76, bottom=0.10, hspace=0.10, wspace=0.05)
grid = np.linspace(*XLIM, 500)
for c, (title, sets) in enumerate(COLS):
    for r, (tag, rowname, who) in enumerate(ROWS):
        ax = axes[r, c]
        ax.axvspan(XLIM[0], 1.0, color="#e9f3ea", zorder=0)
        ax.axvline(1.0, color="0.25", ls="--", lw=0.8, zorder=1)
        ax.text(0.025, 0.95, "inside", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.4, color="#2e6b35", fontweight="bold")
        ymax = 0
        for i, (key, label, col) in enumerate(sets):
            s_ = np.clip(d[f"{key}_{tag}"], None, XLIM[1] * 1.5)
            y = gaussian_kde(s_)(grid)
            ymax = max(ymax, y.max())
            ax.plot(grid, y, color=col, lw=1.2, label=label, zorder=3)
            ax.fill_between(grid[grid <= 1], y[grid <= 1], color=col, alpha=0.16, lw=0, zorder=2)
            ax.text(0.025, 0.95 - 0.12 * (i + 1), f"{100 * (d[f'{key}_{tag}'] <= 1).mean():.1f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=col, fontweight="bold")
        ax.text(0.985, 0.95, "outside", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.4, color="0.35", fontweight="bold")
        ax.set_xlim(*XLIM); ax.set_ylim(0, ymax * 1.1); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"{rowname}\n{who}", labelpad=3)
    top = axes[0, c]
    top.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
               handlelength=1.5, columnspacing=1.0, borderaxespad=0, labelspacing=0.25)
    top.text(0, 1.42, title, transform=top.transAxes, ha="left", va="bottom", fontsize=7.8)
fig.text(0.535, 0.012, "distance to the nearest image of the other set, relative to its neighbourhood radius",
         ha="center", va="bottom", fontsize=7)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT + ".pdf", "XLIM", XLIM)
