"""Outlier rate per measure: how often does a model produce an image that lies OUTSIDE the
range real radiographs occupy?

The means in anatomy.py describe a typical sample; a generator can match them and
still produce a minority of samples that no real radiograph resembles. That is a
tail question, so measure the tail: for each measure, take the interval spanned by
the central 95% of the REAL images (2.5th to 97.5th percentile) and report the
fraction of each model's images falling outside it. A generator that matches the
real distribution scores the same as the real row.

Three measures are used, as in the paper: bilateral symmetry, tooth-row periodicity
and occlusal separation. anatomy.py also stores a crown count (n_crowns); it is not
reported in the paper and is not used here. The column ANY (outside on any measure)
is the last column of the anatomy table of the paper (Table 11), which
make_anat_table.py writes.

Reads out/anat_*.json in the current folder (written by anatomy.py).
"""
import glob, json, os
import numpy as np

OUT = "out"
CENTRAL = 95                                   # central range of the real images, in percent
TAIL = (100 - CENTRAL) / 2
KEYS = ["symmetry", "periodicity", "occlusal"]
d = {}
for p in glob.glob(os.path.join(OUT, "anat_*.json")):
    j = json.load(open(p)); d[j["arm"]] = j

real = {k: np.array(d["real"][k], dtype=float) for k in KEYS}
lo = {k: np.percentile(real[k], TAIL) for k in KEYS}
hi = {k: np.percentile(real[k], 100 - TAIL) for k in KEYS}
print(f"real [p{TAIL:g}, p{100 - TAIL:g}] intervals:")
for k in KEYS:
    print(f"  {k:14s} [{lo[k]:8.3f}, {hi[k]:8.3f}]")

order = ["real", "PanoDiffSR_existing", "FastGAN_HATSR_existing", "sg2ada", "sg3",
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
