"""Statistics of the MedSAM probe: do real and synthetic radiographs behave the same under MedSAM?

Reads out/probe_shard*.csv (one row per prompt) and writes out/medsam_stats.json + a readable
out/medsam_stats.md. For every protocol and measure:
  - mean, sd and median for real and for synthetic;
  - Mann-Whitney U (two-sided) with the rank-biserial effect size r = 2*AUC - 1, which is the
    quantity to read: with thousands of images almost any difference is "significant", so the
    size of the effect is what matters (|r| < 0.1 negligible, 0.1-0.3 small, > 0.3 medium+);
  - the two-sample KS statistic, a distribution-level distance in [0, 1].
Also per source dataset for the real images, so a synthetic-vs-real gap can be compared with the
real-vs-real gaps between devices, and MedSAM's Dice against manual ground truth on real images
(the check that MedSAM segments panoramic teeth at all).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import glob, json, csv, collections, sys
import numpy as np
from scipy import stats

E = WORK + "/medsam-probe"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{E}/out"   # which probe run to summarise
rows = []
for p in sorted(glob.glob(f"{OUT}/probe_shard*.csv")):
    rows += list(csv.DictReader(open(p)))
MEAS = ["pred_iou", "area", "n_comp", "solidity", "dice_ref"]


def col(sel, m):
    return np.array([float(r[m]) for r in sel if r[m] != ""])


def compare(a, b):
    if len(a) < 5 or len(b) < 5:
        return None
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = u.statistic / (len(a) * len(b))
    return {"n_real": len(a), "n_syn": len(b), "real_mean": float(a.mean()), "real_sd": float(a.std(ddof=1)),
            "real_median": float(np.median(a)), "syn_mean": float(b.mean()), "syn_sd": float(b.std(ddof=1)),
            "syn_median": float(np.median(b)), "p": float(u.pvalue), "rank_biserial": float(2 * auc - 1),
            "ks": float(stats.ks_2samp(a, b).statistic)}


out = {"n_rows": len(rows), "protocols": {}, "per_source": {}, "gt_check": {}, "teeth_per_image": {}}
for proto in ("fixed", "arch", "teeth"):
    sel = [r for r in rows if r["protocol"] == proto]
    real = [r for r in sel if r["set"] == "real"]; syn = [r for r in sel if r["set"] == "syn"]
    out["protocols"][proto] = {m: compare(col(real, m), col(syn, m)) for m in MEAS}
    # real-vs-real: each source device against the other real devices, same measures
    for src in sorted({r["source"] for r in real}):
        a = [r for r in real if r["source"] == src]; b = [r for r in real if r["source"] != src]
        out["per_source"].setdefault(proto, {})[src] = {m: compare(col(a, m), col(b, m)) for m in ("pred_iou", "dice_ref")}
    g = col(real, "dice_gt")
    out["gt_check"][proto] = {"dice_gt_mean": float(g.mean()), "dice_gt_median": float(np.median(g)), "n": len(g)} if len(g) else None
# how many teeth the reference U-Net gave MedSAM per image, real vs synthetic
cnt = collections.Counter((r["set"], r["idx"]) for r in rows if r["protocol"] == "teeth")
tr = np.array([v for (s, _), v in cnt.items() if s == "real"]); ts = np.array([v for (s, _), v in cnt.items() if s == "syn"])
out["teeth_per_image"] = compare(tr, ts)
json.dump(out, open(f"{OUT}/medsam_stats.json", "w"), indent=2)

L = ["# MedSAM probe: real vs PanoDiff-SR", "", f"rows: {len(rows)}", ""]
L += ["| protocol | measure | real mean (sd) | synthetic mean (sd) | rank-biserial r | KS | p |", "|---|---|---|---|---|---|---|"]
for proto, d in out["protocols"].items():
    for m, c in d.items():
        if c:
            L.append(f"| {proto} | {m} | {c['real_mean']:.3f} ({c['real_sd']:.3f}) | {c['syn_mean']:.3f} ({c['syn_sd']:.3f}) | {c['rank_biserial']:+.3f} | {c['ks']:.3f} | {c['p']:.2g} |")
c = out["teeth_per_image"]
if c:
    L.append(f"| teeth | boxes per image | {c['real_mean']:.2f} ({c['real_sd']:.2f}) | {c['syn_mean']:.2f} ({c['syn_sd']:.2f}) | {c['rank_biserial']:+.3f} | {c['ks']:.3f} | {c['p']:.2g} |")
L += ["", "## Real device vs the other real devices (same measures, for scale)", "",
      "| protocol | device | measure | device mean | others mean | rank-biserial r | KS |", "|---|---|---|---|---|---|---|"]
for proto, d in out["per_source"].items():
    for src, dd in d.items():
        for m, c in dd.items():
            if c:
                L.append(f"| {proto} | {src} | {m} | {c['real_mean']:.3f} | {c['syn_mean']:.3f} | {c['rank_biserial']:+.3f} | {c['ks']:.3f} |")
L += ["", "## MedSAM against manual ground truth (real images only)", ""]
for proto, c in out["gt_check"].items():
    if c:
        L.append(f"- {proto}: Dice vs GT mean {c['dice_gt_mean']:.3f}, median {c['dice_gt_median']:.3f} (n = {c['n']})")
open(f"{OUT}/medsam_stats.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
