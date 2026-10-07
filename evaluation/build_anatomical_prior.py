"""Fixed anatomical regions used by the attention statistics (Section 4.3, Figure 14) and the
anatomical measures (Table 12): population-level maps of where the teeth and the jaws lie in a
panoramic radiograph, from the TUFTS dataset's expert segmentations.

Per-image segmentations do not exist for synthetic radiographs, so a per-image mask is not
available. Averaging the TUFTS expert masks of the teeth and of the maxillomandibular region,
each resized to 1024 x 512, gives one probability map per structure; empty masks are skipped,
so the teeth map averages the 968 non-empty teeth masks and the jaw map all 1000 jaw masks.
The maps are applied identically to real and synthetic images, so any imprecision in them
cannot favour one set. They are averaged in TUFTS's own framing (no corpus crop).

Thresholds are set where the maps are used: 0.5 for both templates in the attention
statistics (evaluation/vit/vit_attention.py, figures/make_attention_process.py); teeth > 0.35
and jaw > 0.5 for the anatomical measures (evaluation/anatomy_memorisation/anatomy.py).

Writes $PANODIFF_WORK/temp-codes/priors/anatomical_priors.npz (keys teeth, jaw, background)
and one PNG per map. Run once, before the scripts above.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os

import numpy as np
from PIL import Image

OD = (WORK + "/project-files/PanoDiff/GT-PanoDiff-Training/Original_Datasets/"
      "TUFTS - Tufts Panoramic Dataset/Tufts Database")
OUT = WORK + "/temp-codes/priors"
W, H = 1024, 512

REGIONS = {"teeth": os.path.join(OD, "Segmentation", "teeth_mask"),
           "jaw": os.path.join(OD, "Segmentation", "maxillomandibular")}


def average_masks(folder):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    acc = np.zeros((H, W), dtype=np.float64)
    n = 0
    for f in files:
        m = Image.open(os.path.join(folder, f)).convert("L").resize(
            (W, H), Image.BILINEAR)
        a = np.asarray(m, dtype=np.float64) / 255.0
        if a.max() > 0:                      # 32 TUFTS teeth masks are empty
            acc += (a > 0.5).astype(np.float64)
            n += 1
    return acc / max(n, 1), n


os.makedirs(OUT, exist_ok=True)
priors = {}
for name, folder in REGIONS.items():
    p, n = average_masks(folder)
    priors[name] = p
    print(f"{name:6}: averaged {n} masks | coverage at p>0.5 = "
          f"{(p > 0.5).mean():.3f} of the frame, "
          f"mean p = {p.mean():.3f}, max p = {p.max():.3f}")
    Image.fromarray((p * 255).astype(np.uint8)).save(
        os.path.join(OUT, f"prior_{name}.png"))

# background = everything the jaw template does not cover
bg = 1.0 - (priors["jaw"] > 0.5).astype(np.float64)
priors["background"] = bg
print(f"background: {bg.mean():.3f} of the frame")

np.savez_compressed(os.path.join(OUT, "anatomical_priors.npz"), **priors)
print(f"\nwrote {OUT}/anatomical_priors.npz")
