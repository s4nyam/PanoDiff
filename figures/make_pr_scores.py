"""Figure for Section 4.5.1: what precision (fidelity) and recall (coverage) in Table 9 measure.

Reads the per-image scores from swinir-ft-eval/code/pr_ratios.py. For every generated image,
the fidelity score is its distance to the nearest real image divided by that real image's
neighbourhood radius (distance to its 3rd nearest real neighbour); for every real image, the
coverage score is the same with the roles swapped. A score <= 1 means "inside", so the share of
each curve left of the dashed line is exactly the precision or recall of Table 9.

Two rows (fidelity, coverage) x two columns (low resolution, full resolution); colours follow
Figure 16 (make_tsne_arms.py).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

SRC = WORK + "/swinir-ft-eval/out/pr_ratios.npz"
OUT = WORK + "/new-files/overleaf/figures/fig22-pr-scores/pr_scores"

COLS = [
    ("(a) Low resolution ($256\\times128$)",
     [("1-2", "PanoDiff (2)", "#d1495b"), ("1-3", "FastGAN (3)", "#E69F00")]),
    ("(b) Full resolution ($1024\\times512$)",
     [("10-4", "PanoDiff + HAT-SR (4)", "#d1495b"), ("10-6ft", "PanoDiff + SwinIR (6)", "#CC79A7"),
      ("10-5", "FastGAN + HAT-SR (5)", "#E69F00"), ("10-7ft", "FastGAN + SwinIR (7)", "#56B4E9")]),
]
ROWS = [("fid", "Fidelity (precision)", "each generated image"),
        ("cov", "Coverage (recall)", "each real image")]
XLIM = (0.82, 1.45)

PT = 1 / 72.27
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.8, "axes.labelsize": 7, "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})

d = np.load(SRC)
fig, axes = plt.subplots(2, 2, figsize=(390 * PT, 262 * PT), sharex=True)
fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.105, hspace=0.10, wspace=0.05)
grid = np.linspace(*XLIM, 400)
for c, (title, sets) in enumerate(COLS):
    for r, (tag, rowname, who) in enumerate(ROWS):
        ax = axes[r, c]
        ax.axvspan(XLIM[0], 1.0, color="#e9f3ea", zorder=0)
        ax.axvline(1.0, color="0.25", ls="--", lw=0.8, zorder=1)
        ymax = 0
        # Share inside, printed in the "inside" zone in the colour of its curve.
        ax.text(0.025, 0.93, "inside", transform=ax.transAxes, ha="left", va="top",
                fontsize=6.4, color="#2e6b35", fontweight="bold")
        for i, (key, label, col) in enumerate(sets):
            s_ = d[f"{key}_{tag}"]
            y = gaussian_kde(s_)(grid)
            ymax = max(ymax, y.max())
            ax.plot(grid, y, color=col, lw=1.3, label=label, zorder=3)
            ax.fill_between(grid[grid <= 1], y[grid <= 1], color=col, alpha=0.18, lw=0, zorder=2)
            ax.text(0.025, 0.93 - 0.135 * (i + 1), f"{100 * (s_ <= 1).mean():.1f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=col,
                    fontweight="bold")
        ax.text(0.985, 0.93, "outside", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.4, color="0.35", fontweight="bold")
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, ymax * 1.1)
        ax.set_yticks([])
        ax.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        if c == 0:
            ax.set_ylabel(f"{rowname}\n{who}", labelpad=3)
    top = axes[0, c]
    top.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
               handlelength=1.5, columnspacing=1.0, borderaxespad=0, labelspacing=0.25)
    top.text(0, 1.30, title, transform=top.transAxes, ha="left", va="bottom",
             fontsize=7.8)
fig.text(0.535, 0.012, "distance to the nearest image of the other set, relative to its neighbourhood radius",
         ha="center", va="bottom", fontsize=7)
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT + ".pdf")
