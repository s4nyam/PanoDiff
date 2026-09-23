"""Held-out real PRs for Table 10, prepared exactly as the training corpus was.

Same 185 DENTEX validation/test radiographs as prep_heldout.py (the 115 that are pixel-identical
to DENTEX training images are excluded), but with the corpus preprocessing applied: every source
radiograph was cropped by (top 64, left 127, bottom 90, right 127) pixels and then resized to
1024x512 with PIL LANCZOS before it entered the 7243-image corpus. The crop was recovered from
the corpus itself: for sampled originals of all five datasets, that crop reproduces the corpus
file (normalised-thumbnail similarity > 0.999 for 50/50 images, mean absolute pixel difference
0.5-3.5 grey levels, the rest being the resampling implementation).

Writes heldout_crop/{HR,LR} and heldout_crop/heldout_list.json.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
E = WORK + "/sr-restore-7243"
SRC, OUT = f"{E}/heldout", f"{E}/heldout_crop"
CROP = (64, 127, 90, 127)          # top, left, bottom, right, in original pixels

names = json.load(open(f"{SRC}/heldout_list.json"))["names"]
d = json.load(open(f"{SRC}/dup_check.json"))
keep = [c for c, s in zip(d["cands"], d["max_sim"]) if s <= 0.98]
os.makedirs(f"{OUT}/HR", exist_ok=True); os.makedirs(f"{OUT}/LR", exist_ok=True)
out = []
for c in keep:
    n = (("val_" if c.startswith("validation") else "") + os.path.basename(c)).replace("val_val_", "val_")
    im = Image.open(f"{SRC}/dl/{c}").convert("L")
    w, h = im.size
    t, l, b, r = CROP
    g = im.crop((l, t, w - r, h - b)).resize((1024, 512), Image.LANCZOS)
    hr = np.repeat(np.asarray(g)[..., None], 3, axis=2)
    lr = np.rint(hr.astype(np.float64).reshape(128, 4, 256, 4, 3).mean((1, 3))).astype(np.uint8)
    Image.fromarray(hr).save(f"{OUT}/HR/{n}"); Image.fromarray(lr).save(f"{OUT}/LR/{n}")
    out.append(n)
assert sorted(out) == sorted(names), (len(out), len(names))
json.dump({"n": len(out), "names": sorted(out), "crop_top_left_bottom_right": CROP,
           "excluded_duplicates": len(d["cands"]) - len(keep)}, open(f"{OUT}/heldout_list.json", "w"), indent=1)
print("wrote", len(out), "pairs to", OUT)
