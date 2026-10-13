"""How often does a block of horizontal streaks at the left or right image border occur?

A few real radiographs of the corpus carry such a block (rows of pixels smeared sideways over
roughly the outer fifth of the image). The latent diffusion model reproduces it in about half of
its samples. The examples of the direct high-resolution figure were chosen without it, and the
caption of that figure says how often it occurs. This script counts the block in every image set
of the anatomical table, on ALL images of each set.

Measure, per image (grayscale, resized to 512 x 256): in the outer eighth of the width on each
side, the mean absolute difference between vertically adjacent pixels divided by the mean
absolute difference between horizontally adjacent pixels. Ordinary anatomy gives a ratio near 1;
horizontal streaks change from row to row but hardly along a row, which gives 3 to 6. An image
counts as having a block when the larger of its two ratios exceeds THRESHOLD.

The ratios of the LDM samples fall into two groups (below 1.5, and above 3.5) with about 8% of
the samples in between, so its count depends little on the threshold; the counts for 2.0 and 3.0
are stored next to the one for THRESHOLD. Checked by eye on contact sheets (2026-10-13): the
flagged real, PanoDiff-SR and LDM images show the block. A few unflagged images show a narrow or
partial block (about one in ten of the unflagged LDM images looked at), so the counts are lower
bounds. The measure is not specific outside these sets: in StyleGAN3 it flags images that are
smeared as a whole, a different failure, so only the real, PanoDiff-SR and LDM counts are quoted
in the paper.

Writes out/border_streaks.json (summary and the ratio of every image).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
from multiprocessing import Pool
import numpy as np
from PIL import Image

B = WORK + "/experiments/direct-hr-baselines"
ARMS = [("real", "refs/real"), ("PanoDiffSR_existing", "refs/PanoDiffSR_existing"),
        ("sg2ada", "samples/sg2ada"), ("sg3", "samples/sg3"), ("adm", "samples/adm"),
        ("ldm", "samples/ldm"), ("panodiff_hr", "samples/panodiff_hr")]
W, H, STRIP = 512, 256, 64
THRESHOLD = 2.5


def ratios(path):
    a = np.asarray(Image.open(path).convert("L").resize((W, H), Image.BILINEAR), dtype=np.float32)
    out = []
    for strip in (a[:, :STRIP], a[:, -STRIP:]):
        gy = np.abs(np.diff(strip, axis=0)).mean()
        gx = np.abs(np.diff(strip, axis=1)).mean()
        out.append(float(gy / (gx + 1e-6)))
    return out


def main():
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    record = {"measure": "max over the left and right border strip (one eighth of the width) of "
                         "mean|vertical difference| / mean|horizontal difference|, image at 512x256",
              "threshold": THRESHOLD, "sets": {}}
    for name, rel in ARMS:
        folder = os.path.join(B, rel)
        fs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        with Pool(ncpu) as p:
            rows = np.array(p.map(ratios, [os.path.join(folder, f) for f in fs], chunksize=32))
        r = rows.max(axis=1)
        record["sets"][name] = {
            "n": len(fs),
            "with_block": int((r > THRESHOLD).sum()),
            "percent": round(100 * float((r > THRESHOLD).mean()), 2),
            "percent_other_thresholds": {str(t): round(100 * float((r > t).mean()), 2) for t in (2.0, 3.0)},
            "on_the_left": int((rows[:, 0] > THRESHOLD).sum()),
            "ratio": dict(zip(fs, [round(float(x), 3) for x in r])),
        }
        s = record["sets"][name]
        print(f"{name:20s} n={s['n']}  with block {s['with_block']:5d} = {s['percent']:5.2f}%  "
              f"(threshold 2.0: {s['percent_other_thresholds']['2.0']:.2f}%, 3.0: {s['percent_other_thresholds']['3.0']:.2f}%)",
              flush=True)
    out = os.path.join(B, "analysis_r25_r31/out", "border_streaks.json")
    json.dump(record, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
