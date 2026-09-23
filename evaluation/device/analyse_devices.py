"""
Device-stratified FID (reviewer R4 Major 2a).

The reviewer's concern is that pooling five devices and resizing to a common
frame lets the model learn device artefacts rather than anatomy, and that FID
against the pooled corpus is therefore hard to interpret. This measures the
effect directly: how far the synthetic set sits from each source device's own
distribution, and how far the devices sit from each other.

Two things make this harder than it looks and are handled explicitly:

  * FID is strongly biased upward at small sample size, and the sources differ
    by more than 7x in size (ADLD 500 vs DENTEX 3653). Comparing raw per-source
    FIDs would mostly measure sample size. Every comparison here is therefore
    made at a matched n, drawn repeatedly, and reported as mean +- sd over
    draws.
  * The bias does not cancel between a small and a large set, so the reference
    set is subsampled to the same n as the source.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import glob
import json
import os

import numpy as np

from fid_fast import frechet_fast, stats_from

T = WORK + "/temp-codes"
N_DRAWS = 5


def matched_fid(fa, fb, n, draws=N_DRAWS, seed=0):
    """FID between two feature sets, both subsampled to n, averaged over draws."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        a = fa[rng.choice(len(fa), n, replace=False)]
        b = fb[rng.choice(len(fb), n, replace=False)]
        vals.append(frechet_fast(*stats_from(a), *stats_from(b)))
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------- load ------
src = {}
SRC = os.environ.get("SRC_DIR", "stats_src")          # stats_src_crop = corpus preprocessing
OUT_JSON = os.environ.get("OUT_JSON", "device_stratified.json")
for f in sorted(glob.glob(os.path.join(T, SRC, "src-*.npz"))):
    d = np.load(f, allow_pickle=True)
    name = str(d["name"]).replace("src-", "")
    if "features" not in d:
        print(f"  {name}: no cached features, skipping")
        continue
    src[name] = d["features"].astype(np.float64)
    print(f"  {name}: {src[name].shape[0]} images")

gen = {}
for tag, fn in [("PD-HR", "4-DiffHATSR"), ("GT-HR", "10-TrainDataInHR"),
                ("GAN-HR", "5-GANsHATSR")]:
    for d in (os.path.join(T, "stats"), os.path.join(T, "stats_orderedsplit")):
        p = os.path.join(d, fn + ".npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            if "features" in z:
                gen[tag] = z["features"].astype(np.float64)
                print(f"  {tag}: {gen[tag].shape[0]} images (features)")
                break
    else:
        print(f"  {tag}: features not available yet")

if not src or "PD-HR" not in gen:
    raise SystemExit("need per-image features for both the sources and PD-HR; "
                     "re-run the embedding jobs with --save-features")

n_match = min(min(v.shape[0] for v in src.values()),
              min(v.shape[0] for v in gen.values()))
print(f"\nmatched sample size n = {n_match} "
      f"(set by the smallest source), {N_DRAWS} draws per comparison\n")

# ------------------------------------------------ synthetic vs each device --
print("=" * 84)
print("FID FROM EACH SOURCE DEVICE TO THE SYNTHETIC SET (matched n)")
print("=" * 84)
print(f"{'source':10} {'n':>6} {'vs PD-HR':>16} {'vs GT-HR (pooled real)':>24}")
rows = []
for name in sorted(src):
    m_pd, s_pd = matched_fid(src[name], gen["PD-HR"], n_match)
    m_gt, s_gt = matched_fid(src[name], gen["GT-HR"], n_match)
    rows.append(dict(source=name, n=int(src[name].shape[0]),
                     fid_pd=m_pd, sd_pd=s_pd, fid_gt=m_gt, sd_gt=s_gt))
    print(f"{name:10} {src[name].shape[0]:>6} "
          f"{m_pd:9.2f} +- {s_pd:4.2f} {m_gt:16.2f} +- {s_gt:4.2f}")

v = np.array([r["fid_pd"] for r in rows])
print(f"\nspread across devices: {v.min():.1f} to {v.max():.1f} "
      f"(ratio {v.max()/v.min():.2f}x), sd {v.std():.1f}")

# ----------------------------------------------------- device vs device -----
print("\n" + "=" * 84)
print("FID BETWEEN SOURCE DEVICES (matched n) -- the heterogeneity of the corpus")
print("=" * 84)
names = sorted(src)
pair_rows = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        m, s = matched_fid(src[a], src[b], n_match)
        pair_rows.append(dict(a=a, b=b, fid=m, sd=s))
        print(f"  {a:10} vs {b:10} {m:7.2f} +- {s:4.2f}")
pv = np.array([r["fid"] for r in pair_rows])
print(f"\ndevice-to-device FID ranges {pv.min():.1f}-{pv.max():.1f}, mean {pv.mean():.1f}")

# ------------------------------------------------------------- reference ----
m_self, s_self = matched_fid(gen["GT-HR"], gen["GT-HR"], n_match // 2, seed=1)
print(f"\nreference floor: two disjoint draws from the pooled real set at "
      f"n={n_match//2} give FID {m_self:.2f} +- {s_self:.2f}")

print("\n" + "=" * 84)
print("READ-OUT")
print("=" * 84)
print(f"The synthetic set sits {v.min():.0f}-{v.max():.0f} from the individual")
print(f"devices, while the devices sit {pv.min():.0f}-{pv.max():.0f} from each other.")
print("Whether the generator favours particular devices is judged by comparing")
print("the spread of the first row against the spread of the second.")

json.dump({"synthetic_vs_device": rows, "device_vs_device": pair_rows,
           "matched_n": int(n_match), "draws": N_DRAWS,
           "floor": {"mean": m_self, "sd": s_self}},
          open(os.path.join(T, OUT_JSON), "w"), indent=2)
print(f"\nwrote {OUT_JSON}")
