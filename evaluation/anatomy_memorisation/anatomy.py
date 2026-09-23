"""Anatomical plausibility measures for synthetic panoramic radiographs.

A clinician reading these images reports that the StyleGAN2-ADA samples, despite
their low FID, contain implausible constructions -- jaw outlines that do not
close, tooth rows that do not alternate regularly, crowns without roots. FID
cannot see any of that: it compares pooled Inception statistics over a whole set
and is insensitive to whether an individual image is anatomically coherent. This
script turns that clinical reading into four measurements that a reviewer can
check, each chosen because it is a property every real PR has and a generator can
plausibly fail to reproduce.

  symmetry     Pearson r between the jaw-masked image and its mirror. A PR images
               both hemi-arches in one projection, so it is approximately
               symmetric about the midline.
  periodicity  Fraction of spectral power of the column-intensity profile of the
               tooth band that falls in the tooth-spacing band (period 20-60 px
               at 1024 width). Teeth form a quasi-periodic row; noise and blur
               both destroy this, in opposite directions.
  n_crowns     Connected components of high-intensity enamel inside the tooth
               prior, with a plausible area range. Proxy for how many distinct
               crowns are actually rendered.
  occlusal     Depth of the dark occlusal line separating the upper and lower
               tooth rows, relative to the two bright rows around it.

Every arm is scored, INCLUDING the ones that failed outright (Medfusion, the two
FastGAN runs). Those are negative controls: a measure that does not separate pure
noise from a real radiograph is not measuring anatomy, and should not be believed
when it is applied to a close case.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage, stats

B = WORK + "/new-baselines-generation"
T = WORK + "/temp-codes"
ARMS = [("real", "refs/real"), ("PanoDiffSR_existing", "refs/PanoDiffSR_existing"),
        ("FastGAN_HATSR_existing", "refs/FastGAN_HATSR_existing"),
        ("sg2ada", "samples/sg2ada"), ("adm", "samples/adm"), ("ldm", "samples/ldm"),
        ("panodiff_hr", "samples/panodiff_hr"), ("fastgan_lite", "samples/fastgan_lite"),
        ("medfusion", "samples/medfusion"), ("fastgan_hr", "samples/fastgan_hr"),
        ("sg3", "samples/sg3")]
if os.environ.get("ONLY_ARM"):
    ARMS = [a for a in ARMS if a[0] == os.environ["ONLY_ARM"]]
N_PER_ARM = 1500
H, W = 512, 1024

_P = np.load(os.path.join(T, "priors", "anatomical_priors.npz"))
TEETH = _P["teeth"] > 0.35
JAW = _P["jaw"] > 0.5
_rows = np.where(TEETH.any(axis=1))[0]
R0, R1 = int(_rows.min()), int(_rows.max())
_cols = np.where(TEETH.any(axis=0))[0]
C0, C1 = int(_cols.min()), int(_cols.max())


def measure(path):
    a = np.asarray(Image.open(path).convert("L").resize((W, H), Image.BILINEAR),
                   dtype=np.float32)

    # ---- symmetry: correlation of the jaw region with its mirror image
    m = JAW
    x, y = a[m], np.fliplr(a)[m]
    sym = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else 0.0

    band = a[R0:R1 + 1, C0:C1 + 1]

    # ---- periodicity of the tooth row along the arch
    prof = band.mean(axis=0)
    prof = prof - prof.mean()
    if prof.std() > 1e-6:
        prof = prof / prof.std()
        pw = np.abs(np.fft.rfft(prof * np.hanning(len(prof)))) ** 2
        freqs = np.fft.rfftfreq(len(prof), d=1.0)
        with np.errstate(divide="ignore"):
            period = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-9), np.inf)
        sel = (period >= 20) & (period <= 60)
        tot = pw[1:].sum()
        periodicity = float(pw[sel].sum() / tot) if tot > 0 else 0.0
    else:
        periodicity = 0.0

    # ---- crown count: bright enamel blobs inside the tooth prior
    tb = TEETH[R0:R1 + 1, C0:C1 + 1]
    vals = band[tb]
    thr = np.percentile(vals, 88) if vals.size else 255.0
    mask = (band > thr) & tb
    lab, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        n_crowns = int(((sizes >= 60) & (sizes <= 4000)).sum())
    else:
        n_crowns = 0

    # ---- occlusal line: dark separation between the two tooth rows
    rp = band.mean(axis=1)
    if len(rp) >= 9 and rp.std() > 1e-6:
        q = len(rp) // 4
        upper = rp[:q].max() if q else rp.max()
        lower = rp[-q:].max() if q else rp.max()
        mid = rp[len(rp) // 3: 2 * len(rp) // 3]
        occl = float((min(upper, lower) - mid.min()) / (rp.max() - rp.min() + 1e-6))
    else:
        occl = 0.0
    return sym, periodicity, n_crowns, occl


def main():
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    mine = [ARMS[i] for i in range(len(ARMS)) if i % world == rank]
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    for name, rel in mine:
        folder = os.path.join(B, rel)
        fs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        rng = np.random.default_rng(0)
        pick = [fs[i] for i in rng.permutation(len(fs))[:N_PER_ARM]]
        paths = [os.path.join(folder, f) for f in pick]
        with Pool(ncpu) as p:
            rows = p.map(measure, paths, chunksize=8)
        arr = np.array(rows, dtype=np.float64)
        out = dict(arm=name, n=len(rows),
                   symmetry=arr[:, 0].tolist(), periodicity=arr[:, 1].tolist(),
                   n_crowns=arr[:, 2].tolist(), occlusal=arr[:, 3].tolist())
        json.dump(out, open(os.path.join(B, "analysis_r25_r31/out", f"anat_{name}.json"), "w"))
        print(f"{name:24s} n={len(rows):5d} sym {arr[:,0].mean():6.3f} "
              f"per {arr[:,1].mean():6.3f} crowns {arr[:,2].mean():6.2f} "
              f"occl {arr[:,3].mean():6.3f}", flush=True)


if __name__ == "__main__":
    main()
