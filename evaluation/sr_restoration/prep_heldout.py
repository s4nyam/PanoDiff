"""Held-out real PRs for the restoration test: DENTEX validation (50) and test (250) radiographs that are not in
the training corpus.

Every real PR we hold went into the 7243-image corpus on which both SR models were fine-tuned,
so held-out images have to come from outside it. DENTEX's official validation and test sets were
downloaded from huggingface.co/datasets/ibrahimhamamci/DENTEX (DENTEX/validation_data.zip,
DENTEX/test_data.zip) into heldout/dl. 115 of the 300 are exact copies of images in DENTEX's
training folders (normalised 64x32 thumbnail similarity 1.000 to a training original, which the
corpus contains); the next closest scores 0.921 (0.910 at 256x128), within the range of distinct
DENTEX radiographs (median nearest-neighbour similarity among them 0.89). heldout/dup_check.json
holds the similarities; images above 0.98 are excluded, leaving 185.

Preparation replicates the corpus: grey-level radiograph as 3 equal channels, LANCZOS resize to
1024x512 (HR, the reference), then the fourfold area average of create_lr.py (cv2.INTER_AREA,
here a 4x4 block mean rounded half-to-even, which reproduces set 1 from set 10 exactly) (LR).
The corpus images were also cropped to the jaws; no crop is applied here. prep_heldout_crop.py
applies the corpus crop to the same radiographs; the LPIPS values of Section 3.4 come from that
version.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
H = WORK + "/experiments/sr-restoration/heldout"
THR = 0.98

d = json.load(open(f"{H}/dup_check.json"))
keep = [c for c, s in zip(d["cands"], d["max_sim"]) if s <= THR]
print(f"{len(d['cands'])} candidates, {len(d['cands']) - len(keep)} duplicates of training images, {len(keep)} held out")
os.makedirs(f"{H}/HR", exist_ok=True); os.makedirs(f"{H}/LR", exist_ok=True)
names = []
for c in keep:
    n = ("val_" if c.startswith("validation") else "") + os.path.basename(c)   # test_12.png / val_val_5.png -> unique
    n = n.replace("val_val_", "val_")
    g = Image.open(f"{H}/dl/{c}").convert("L").resize((1024, 512), Image.LANCZOS)
    hr = np.repeat(np.asarray(g)[..., None], 3, axis=2)
    lr = np.rint(hr.astype(np.float64).reshape(128, 4, 256, 4, 3).mean((1, 3))).astype(np.uint8)
    Image.fromarray(hr).save(f"{H}/HR/{n}"); Image.fromarray(lr).save(f"{H}/LR/{n}")
    names.append(n)
assert len(set(names)) == len(names)
json.dump({"n": len(names), "names": sorted(names), "excluded_duplicates": len(d["cands"]) - len(keep)},
          open(f"{H}/heldout_list.json", "w"), indent=1)
print("wrote", len(names), "pairs")
