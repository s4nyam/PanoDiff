"""Redraw the per-observer prediction-distribution pie charts (manuscript Fig. 7)
at a legible size (reviewer R3-2).

The data, the category definitions, the colour map, the start angle and the
label placement are taken unchanged from the original analysis script
(PanoDiff-QuizData/quiz_fullres_mix/user_wise_results_spreadsheet_backup.py),
so the figure shows exactly the same numbers as the submitted version. The only
changes are layout and type size: the six charts move from a 1x6 strip, which
had to be squeezed into \\linewidth and left the labels at roughly 3 pt on the
page, to a 3x2 grid with correspondingly larger fonts.
"""
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUIZ = sys.argv[1]      # .../PanoDiff-QuizData/quiz_fullres_mix
OUT = sys.argv[2]       # output directory

COLOR_MAP = OrderedDict([
    ("TP_fully", "#4CAF50"), ("TP_partially", "#81C784"),
    ("TN_fully", "#2196F3"), ("TN_partially", "#64B5F6"),
    ("FP_fully", "#F44336"), ("FP_partially", "#E57373"),
    ("FN_fully", "#FF9800"), ("FN_partially", "#FFB74D"),
    ("Unsure",   "#9E9E9E"),
])

# Fractional credit table from the original script.
#   key: (true label, predicted label) -> (TP, TN, FP, FN)
def scores(true, pred):
    tp = tn = fp = fn = 0.0
    if true == "Fake":
        tp = {"Definitely Fake": 1, "Probably Fake": 0.75, "Unsure": 0.5,
              "Probably Real": 0.25, "Definitely Real": 0}.get(pred, 0)
        fn = {"Definitely Real": 1, "Probably Real": 0.75, "Unsure": 0.5,
              "Probably Fake": 0.25, "Definitely Fake": 0}.get(pred, 0)
    elif true == "Real":
        tn = {"Definitely Real": 1, "Probably Real": 0.75, "Unsure": 0.5,
              "Probably Fake": 0.25, "Definitely Fake": 0}.get(pred, 0)
        fp = {"Definitely Fake": 1, "Probably Fake": 0.75, "Unsure": 0.5,
              "Probably Real": 0.25, "Definitely Real": 0}.get(pred, 0)
    return tp, tn, fp, fn


labels = {}
with open(os.path.join(QUIZ, "image_labels.csv")) as fh:
    for r in csv.DictReader(fh):
        labels[r["Filename"].strip()] = r["Label"].strip()

per_user = {}
with open(os.path.join(QUIZ, "submissions.csv")) as fh:
    for r in csv.DictReader(fh):
        u = (r.get("User Name") or "").strip()
        f = (r.get("File") or "").strip()
        p = (r.get("RealFakeValueSlider") or "").strip()
        if not u or f not in labels:          # inner join, as in the original
            continue
        per_user.setdefault(u, []).append((labels[f], p))

pie = {}
for u, rows in per_user.items():
    c = OrderedDict((k, 0) for k in COLOR_MAP)
    for true, pred in rows:
        tp, tn, fp, fn = scores(true, pred)
        if tp == 1:    c["TP_fully"] += 1
        if tp == 0.75: c["TP_partially"] += 1
        if tn == 1:    c["TN_fully"] += 1
        if tn == 0.75: c["TN_partially"] += 1
        if fp == 1:    c["FP_fully"] += 1
        if fp == 0.75: c["FP_partially"] += 1
        if fn == 1:    c["FN_fully"] += 1
        if fn == 0.75: c["FN_partially"] += 1
        if pred == "Unsure": c["Unsure"] += 1
    pie[u] = OrderedDict((k, v) for k, v in c.items() if v > 0)

users = sorted(pie)
print(f"observers: {users}")
for u in users:
    tot = sum(pie[u].values())
    print(f"  {u}  n={tot}  " +
          "  ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k, v in pie[u].items()))

# ----------------------------------------------------------------- figure --
NCOL, NROW = 3, 2
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(NROW, NCOL, figsize=(13.0, 11.0))
axes = axes.ravel()

panels = []          # per-axes label bookkeeping for the de-overlap pass
for ax, user in zip(axes, users):
    data = pie[user]
    sizes = list(data.values())
    colors = [COLOR_MAP[k] for k in data]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=140,
                       wedgeprops={"linewidth": 1, "edgecolor": "white"})
    total = sum(sizes)
    texts, angs, radii, fracs = [], [], [], []
    for w, s in zip(wedges, sizes):
        frac = s / total
        ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
        r = 1.13
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        t = ax.text(r * x, r * y, f"{frac*100:.1f}%", ha="center", va="center",
                    fontsize=18,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.85, edgecolor="none"))
        texts.append(t); angs.append(ang); radii.append(r); fracs.append(frac)
    panels.append((texts, angs, radii, fracs))
    # Give the label ring room inside the axes so nothing collides with the
    # panel title above or the legend below.
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-1.72, 1.72)
    ax.set_title(user, fontsize=23, pad=2)
    ax.set_aspect("equal")

for ax in axes[len(users):]:
    ax.axis("off")

legend_labels = sorted(COLOR_MAP)
fig.legend([plt.Rectangle((0, 0), 1, 1, fc=COLOR_MAP[k]) for k in legend_labels],
           legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
           ncol=5, fontsize=18, title="Metrics", title_fontsize=19,
           frameon=True, framealpha=0.95)
fig.suptitle("Prediction Distribution Across All Dentists", fontsize=24, y=0.985)
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.13,
                    wspace=0.05, hspace=0.16)

# Resolve label collisions on the rendered geometry rather than by guessing an
# angular threshold: repeatedly step the label of the thinner wedge outwards
# until no two label boxes within a panel overlap.
fig.canvas.draw()
rend = fig.canvas.get_renderer()
for _ in range(60):
    clash = False
    for texts, angs, radii, fracs in panels:
        boxes = [t.get_window_extent(rend) for t in texts]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                k = i if fracs[i] < fracs[j] else j      # move the thinner one
                if radii[k] >= 1.60:
                    continue
                clash = True
                radii[k] += 0.09
                a = np.deg2rad(angs[k])
                texts[k].set_position((radii[k] * np.cos(a),
                                       radii[k] * np.sin(a)))
    if not clash:
        break
    fig.canvas.draw()
else:
    print("warning: some labels may still overlap")

os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "piecharts.pdf")
fig.savefig(out, format="pdf")
fig.savefig(out.replace(".pdf", ".png"), dpi=170)
plt.close(fig)
print("wrote", out)
