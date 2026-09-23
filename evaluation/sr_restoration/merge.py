"""Merge the per-rank CSVs into the table: mean and sd per row, plus paired Wilcoxon tests between
rows (for the text; not printed in the table).

    python merge.py              -> the 7243 corpus PRs   (out/parts     -> out/restore_7243.json)
    python merge.py heldout      -> the 185 held-out PRs  (heldout/parts -> heldout/restore_heldout.json)
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import csv, glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

E = WORK + "/sr-restore-7243"
HELD = len(sys.argv) > 1 and sys.argv[1] == "heldout"
OUT = os.environ.get("HELD_DIR", f"{E}/heldout") if HELD else f"{E}/out"
N = 185 if HELD else 7243
ROWS = ["hat_pre", "swin_pre", "hat_ft", "swin_ft"]
M = ["psnr", "ssim", "lpips"]

vals = {r: {} for r in ROWS}
for f in sorted(glob.glob(f"{OUT}/parts/rank_*.csv")):
    for d in csv.DictReader(open(f)):
        vals[d["row"]][d["name"]] = [float(d[m]) for m in M]
names = sorted(vals["hat_ft"])
assert len(names) == N and all(sorted(vals[r]) == names for r in ROWS), {r: len(vals[r]) for r in ROWS}

res = {}
for r in ROWS:
    a = np.array([vals[r][n] for n in names])
    res[r] = {m: dict(mean=float(a[:, i].mean()), sd=float(a[:, i].std())) for i, m in enumerate(M)}
    res[r]["n"] = len(names)
    print(f"{r:9s} " + "  ".join(f"{m} {a[:, i].mean():.4f} ± {a[:, i].std():.4f}" for i, m in enumerate(M)))
for x, y in [("hat_ft", "swin_ft"), ("hat_pre", "swin_pre"), ("hat_pre", "hat_ft"), ("swin_pre", "swin_ft")]:
    for i, m in enumerate(M):
        a = np.array([vals[x][n][i] for n in names]); b = np.array([vals[y][n][i] for n in names])
        p = float(wilcoxon(a, b).pvalue)
        res[f"{x}_vs_{y}_{m}"] = dict(p=p, share_x_better=float(((a < b) if m == "lpips" else (a > b)).mean()))
        print(f"{x} vs {y} {m}: p={p:.2e}, {x} better on {100 * res[f'{x}_vs_{y}_{m}']['share_x_better']:.1f}%")
J = f"{OUT}/restore_heldout.json" if HELD else f"{OUT}/restore_7243.json"
json.dump(res, open(J, "w"), indent=1)
print("wrote", J)
