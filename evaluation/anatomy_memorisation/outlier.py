"""Gross-failure rate: how often does an arm produce an image that lies OUTSIDE
the range real radiographs occupy?

The means in anatomy.py answer "what does a typical sample look like", which is
not the clinical complaint. The complaint is that a substantial minority of
samples contain constructions no real radiograph would show. That is a tail
question, so measure the tail: for each metric, take the [1st, 99th] percentile
interval of the REAL images and report the fraction of each arm falling outside
it. A perfectly calibrated generator would score 2% per metric by construction.
"""
import glob, json, os
import numpy as np

OUT = "out"
KEYS = ["symmetry", "periodicity", "n_crowns", "occlusal"]
d = {}
for p in glob.glob(os.path.join(OUT, "anat_*.json")):
    j = json.load(open(p)); d[j["arm"]] = j

real = {k: np.array(d["real"][k], dtype=float) for k in KEYS}
lo = {k: np.percentile(real[k], 1) for k in KEYS}
hi = {k: np.percentile(real[k], 99) for k in KEYS}
print("real [p01, p99] intervals:")
for k in KEYS:
    print(f"  {k:14s} [{lo[k]:8.3f}, {hi[k]:8.3f}]")

order = ["real", "PanoDiffSR_existing", "FastGAN_HATSR_existing", "sg2ada",
         "adm", "ldm", "panodiff_hr", "fastgan_lite", "fastgan_hr", "medfusion"]
print(f"\n{'arm':24s} " + " ".join(f"{k[:9]:>10s}" for k in KEYS) + f" {'ANY':>8s}")
for arm in order:
    if arm not in d: continue
    a = {k: np.array(d[arm][k], dtype=float) for k in KEYS}
    out = {k: (a[k] < lo[k]) | (a[k] > hi[k]) for k in KEYS}
    anyout = np.zeros(len(a[KEYS[0]]), bool)
    for k in KEYS: anyout |= out[k]
    print(f"{arm:24s} " + " ".join(f"{100*out[k].mean():9.1f}%" for k in KEYS)
          + f" {100*anyout.mean():7.1f}%")
