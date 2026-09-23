"""Anatomy table for the manuscript: per-measure means plus the gross-failure rate.

The gross-failure row is the one that carries the argument, so it is computed here
rather than quoted: for each measure, take the interval spanned by the central 98%
of REAL radiographs and report the fraction of an arm falling outside it on any
measure. A perfectly calibrated generator scores the same as the real row.
"""
import glob, json, os
import numpy as np
from scipy import stats

OUT = "out"
KEYS = [("symmetry", "Bilateral symmetry $r$", 3),
        ("periodicity", "Tooth-row periodicity", 3),
        ("n_crowns", "Distinct enamel crowns", 1),
        ("occlusal", "Occlusal separation", 3)]
COLS = [("real", "Real"), ("PanoDiffSR_existing", "PanoDiff-SR"),
        ("sg2ada", "StyleGAN2-ADA"), ("sg3", "StyleGAN3"), ("adm", "ADM"), ("ldm", "LDM"),
        ("panodiff_hr", "PanoDiff-HR")]

d = {}
for p in glob.glob(os.path.join(OUT, "anat_*.json")):
    j = json.load(open(p)); d[j["arm"]] = j

lo, hi = {}, {}
for k, _, _ in KEYS:
    r = np.array(d["real"][k], float)
    lo[k], hi[k] = np.percentile(r, 1), np.percentile(r, 99)

lines = []
for k, label, dp in KEYS:
    real = np.array(d["real"][k], float)
    cells = []
    for arm, _ in COLS:
        a = np.array(d[arm][k], float)
        if arm == "real":
            cells.append(f"{a.mean():.{dp}f}")
        else:
            _, p = stats.mannwhitneyu(real, a, alternative="two-sided")
            star = "" if p >= 0.05 else ("$^{*}$" if p >= 1e-4 else "$^{**}$")
            cells.append(f"{a.mean():.{dp}f}{star}")
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")

gf = []
for arm, _ in COLS:
    anyout = None
    for k, _, _ in KEYS:
        a = np.array(d[arm][k], float)
        o = (a < lo[k]) | (a > hi[k])
        anyout = o if anyout is None else (anyout | o)
    gf.append(f"{100*anyout.mean():.1f}\\%")
lines.append(r"\midrule")
lines.append(r"\textbf{Outside real range} & " + " & ".join(gf) + r" \\")

hdr = " & ".join(n for _, n in COLS)
print("HEADER: Measure & " + hdr + r" \\")
print("\n".join(lines))
open(os.path.join(OUT, "table_anat_final.tex"), "w").write("\n".join(lines) + "\n")
