"""Every number the MedSAM paragraph of the paper states, from one probe output directory.

    python3 analyze_probe_extra.py <out_dir> [<data_dir>]

Reads <out_dir>/probe_shard*.csv. Prints the table of Table~\\ref{tab:medsam}, the device-vs-others
ranges, MedSAM's Dice against the manual masks, the relative size of the dentition region, and the
size-stratified effect for the dentition box (quintiles of the pooled reference area), which is the
check that the dentition-box gap is not a size effect.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import collections, csv, glob, json, sys
import numpy as np
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else WORK + "/medsam-probe/out"
rows = []
for p in sorted(glob.glob(f"{OUT}/probe_shard*.csv")):
    rows += list(csv.DictReader(open(p)))
print(f"{OUT}: {len(rows)} prompt rows")


def r_of(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u.statistic / (len(a) * len(b)) - 1, u.pvalue


def col(sel, m):
    return np.array([float(r[m]) for r in sel if r[m] != ""])


NAME = {"fixed": "Fixed box", "arch": "Dentition box", "teeth": "Tooth region"}
for proto in ("fixed", "arch", "teeth"):
    sel = [r for r in rows if r["protocol"] == proto]
    real = [r for r in sel if r["set"] == "real"]; syn = [r for r in sel if r["set"] == "syn"]
    print(f"\n== {NAME[proto]}  (real {len(real)}, synthetic {len(syn)} prompts)")
    for m in ("pred_iou", "dice_ref", "area", "n_comp", "solidity"):
        a, b = col(real, m), col(syn, m)
        r, p = r_of(a, b)
        print(f"   {m:9s} real {a.mean():8.3f}  syn {b.mean():8.3f}  r {r:+.2f}  p {p:.1e}")
    g = col(real, "dice_gt")
    if len(g):
        print(f"   dice vs manual masks (real): {g.mean():.3f}")
    # device against the other devices, for scale
    for m in ("pred_iou", "dice_ref"):
        rs = []
        for src in sorted({x["source"] for x in real}):
            a = col([x for x in real if x["source"] == src], m)
            b = col([x for x in real if x["source"] != src], m)
            rs.append(abs(r_of(a, b)[0]))
        print(f"   device vs others |r| ({m}): {min(rs):.2f}-{max(rs):.2f}")

# dentition region size and the size-stratified effect
sel = [r for r in rows if r["protocol"] == "arch"]
ar = np.array([float(r["ref_area"]) for r in sel]); iou = np.array([float(r["pred_iou"]) for r in sel])
isreal = np.array([r["set"] == "real" for r in sel])
print(f"\n== Dentition region: real {ar[isreal].mean():.0f} px, synthetic {ar[~isreal].mean():.0f} px "
      f"({100 * (ar[~isreal].mean() / ar[isreal].mean() - 1):+.0f}%)")
src = np.array([r["source"] for r in sel])
dev = {s: ar[isreal & (src == s)].mean() for s in sorted(set(src[isreal]))}
print("   per device:", {k: int(v) for k, v in dev.items()},
      f"| largest/smallest device ratio {max(dev.values()) / min(dev.values()):.2f}")
q = np.quantile(ar, [0.2, 0.4, 0.6, 0.8])
b = np.digitize(ar, q)
print("   quality-score effect within quintiles of the reference area:")
for k in range(5):
    m = b == k
    a_, s_ = iou[m & isreal], iou[m & ~isreal]
    if len(a_) >= 5 and len(s_) >= 5:
        r, _ = r_of(a_, s_)
        print(f"     Q{k + 1} ({ar[m].min():6.0f}-{ar[m].max():6.0f} px, real {len(a_):4d}, syn {len(s_):4d}): "
              f"real {a_.mean():.3f} syn {s_.mean():.3f}  r {r:+.2f}")

# tooth regions per image
cnt = collections.Counter((r["set"], r["idx"]) for r in rows if r["protocol"] == "teeth")
tr = np.array([v for (s, _), v in cnt.items() if s == "real"]); ts = np.array([v for (s, _), v in cnt.items() if s == "syn"])
print(f"\n== Tooth regions per image: real {tr.mean():.2f}, synthetic {ts.mean():.2f}, r {r_of(tr, ts)[0]:+.2f}")
per_src = {}
for r_ in rows:
    if r_["protocol"] == "teeth" and r_["set"] == "real":
        per_src.setdefault(r_["source"], set()).add(r_["idx"])
cnt_src = {s: np.mean([v for (st, i), v in cnt.items() if st == "real" and i in idxs])
           for s, idxs in per_src.items()}
print("   per device:", {k: round(v, 1) for k, v in sorted(cnt_src.items())})
