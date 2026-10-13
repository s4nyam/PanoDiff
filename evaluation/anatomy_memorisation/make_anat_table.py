"""Anatomy table for the manuscript (Table 11): mean and standard deviation of each measure, and the
outlier rate.

Three measures are reported: bilateral symmetry, tooth-row periodicity and occlusal separation.
anatomy.py also stores a crown count (n_crowns) for every image; it is no longer reported (removed
from the paper on 9 Oct 2026: its intensity percentile is affected by metal in the image).

The outlier rate is computed here rather than quoted: for each measure, take the interval spanned
by the central 95% of REAL radiographs (2.5th to 97.5th percentile, the usual reference interval)
and report the fraction of a model's images falling outside it on any measure. A generator that
matches the real distribution scores the same as the real row. The same rate for the central
98% and 99% is printed as a check that the ranking of the models does not depend on this choice.

Run from the folder that holds out/ (anatomy.py writes out/anat_*.json to
$PANODIFF_WORK/experiments/direct-hr-baselines/analysis_r25_r31/). Reads out/anat_*.json, writes
out/table_anat_final.tex and out/table_anat_final.json.
"""
import glob, json, os
import numpy as np
from scipy import stats

OUT = "out"
CENTRAL = 95
KEYS = [("symmetry", "Bilateral symmetry $r$", 3),
        ("periodicity", "Tooth-row periodicity", 3),
        ("occlusal", "Occlusal separation", 3)]
COLS = [("real", "Real"), ("PanoDiffSR_existing", "PanoDiff-SR"),
        ("sg2ada", "StyleGAN2-ADA"), ("sg3", "StyleGAN3"), ("adm", "ADM"), ("ldm", "LDM"),
        ("panodiff_hr", "PanoDiff-HR")]

d = {}
for p in glob.glob(os.path.join(OUT, "anat_*.json")):
    j = json.load(open(p)); d[j["arm"]] = j


def outside(central):
    """fraction of each model's images outside the central range of the real images on any measure"""
    tail = (100 - central) / 2
    res = {}
    for arm, _ in COLS:
        anyout = None
        for k, _, _ in KEYS:
            r = np.array(d["real"][k], float)
            lo, hi = np.percentile(r, tail), np.percentile(r, 100 - tail)
            a = np.array(d[arm][k], float)
            o = (a < lo) | (a > hi)
            anyout = o if anyout is None else (anyout | o)
        res[arm] = 100 * float(anyout.mean())
    return res


record = {"n_per_model": {arm: int(d[arm]["n"]) for arm, _ in COLS}, "central_range_percent": CENTRAL, "measures": {}}
cell = {}                                  # (measure, model) -> "mean \pm sd" with the significance mark
for k, label, dp in KEYS:
    real = np.array(d["real"][k], float)
    record["measures"][k] = {}
    for arm, _ in COLS:
        a = np.array(d[arm][k], float)
        m, sd = a.mean(), a.std(ddof=1)
        if arm == "real":
            star, p = "", None
        else:
            _, p = stats.mannwhitneyu(real, a, alternative="two-sided")
            star = "" if p >= 0.05 else ("^{*}" if p >= 1e-4 else "^{**}")
        cell[k, arm] = f"${m:.{dp}f} \\pm {sd:.{dp}f}{star}$"
        record["measures"][k][arm] = dict(mean=float(m), sd=float(sd), p_mann_whitney=None if p is None else float(p))

rate = outside(CENTRAL)
record["outside_real_range_percent"] = rate
record["outside_real_range_percent_other_ranges"] = {str(c): outside(c) for c in (98, 99)}

# one row per model, one column per measure (the table is set this way in the manuscript)
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \textbf{Bilateral} & \textbf{Tooth-row} & \textbf{Occlusal} & \textbf{Outside} \\",
         r"\textbf{Images} & \textbf{symmetry $r$} & \textbf{periodicity} & \textbf{separation} & \textbf{real range} \\",
         r"\midrule"]
for arm, name in COLS:
    lines.append(f"{name} & " + " & ".join(cell[k, arm] for k, _, _ in KEYS) + f" & ${rate[arm]:.1f}\\%$ \\\\")
    if arm == "real": lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]

print("\n".join(lines))
print()
for c in (CENTRAL, 98, 99):
    r = outside(c)
    order = sorted((arm for arm, _ in COLS[1:]), key=lambda arm: r[arm])
    print(f"central {c}%: " + "  ".join(f"{arm} {r[arm]:.1f}" for arm, _ in COLS) + "   | ranking: " + " < ".join(order))
open(os.path.join(OUT, "table_anat_final.tex"), "w").write("\n".join(lines) + "\n")
json.dump(record, open(os.path.join(OUT, "table_anat_final.json"), "w"), indent=1)
