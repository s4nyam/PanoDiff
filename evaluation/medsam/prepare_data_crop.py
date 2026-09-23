"""Rebuild the real MedSAM set with the preprocessing the training corpus used.

The real radiographs of the probe were resized to 1024x512 straight from the source files
(prepare_data.py), whereas every radiograph of the 7243-image corpus that PanoDiff and the
upscalers were trained on was first cropped by (top 64, left 127, bottom 90, right 127) pixels
and then resized to 1024x512 with PIL LANCZOS. The synthetic images therefore inherit the
corpus framing and the real ones did not, which biases anything that depends on how much of the
radiograph is in view. This script applies the corpus preprocessing to the real images and to
their manual masks (same crop, nearest-neighbour resize), keeping the loaders, the image order,
the dropped-empty-mask rule and the 70/15/15 split of prepare_data.py unchanged.

The crop was recovered from the corpus: for sampled originals of all five source datasets it
reproduces the matching corpus file (normalised-thumbnail similarity > 0.999, mean absolute
pixel difference 0.5-3.5 grey levels, the remainder being the resampling implementation).

Writes medsam-probe/data_crop/{real_img,real_msk}.npy, meta.json, fractions.json, and links the
unchanged synthetic arrays.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, math, os, sys
import numpy as np
from multiprocessing import Pool
from PIL import Image

sys.path.insert(0, WORK + "/downstream-seg/code")
import prepare_data as pd  # noqa: E402

OUT = WORK + "/medsam-probe/data_crop"
OLD = WORK + "/downstream-seg/data"
CROP = (64, 127, 90, 127)          # top, left, bottom, right, in source pixels
H, W = pd.H, pd.W


def finish(img, msk):
    """Corpus preprocessing: crop the fixed margins, then LANCZOS (image) / NEAREST (mask)."""
    h, w = img.shape[:2]
    t, l, b, r = CROP
    t, b = min(t, max(0, h // 4)), min(b, max(0, h // 4))
    l, r = min(l, max(0, w // 4)), min(r, max(0, w // 4))
    img = img[t:h - b, l:w - r]
    msk = msk.astype(np.uint8)[t:h - b, l:w - r]
    img = np.asarray(Image.fromarray(img).resize((W, H), Image.LANCZOS))
    msk = np.asarray(Image.fromarray(msk).resize((W, H), Image.NEAREST))
    return img.astype(np.uint8), (msk > 0).astype(np.uint8)


pd.finish = finish                                  # the loaders call the module-level name


def main():
    os.makedirs(OUT, exist_ok=True)
    items = list(pd.items_adld()) + list(pd.items_tsxk()) + list(pd.items_tufts()) + list(pd.items_dentex())
    print("real items:", len(items), flush=True)
    with Pool(os.cpu_count()) as pool:
        res = pool.map(pd.work, items, chunksize=8)
    bad = [(r[0], r[1], r[4]) for r in res if r[4]]
    if bad:
        print("FAILED:", len(bad), bad[:5], flush=True)
    res = [r for r in res if r[4] is None and r[3].sum() > 0]
    n = len(res)
    print("kept:", n, flush=True)

    img = np.lib.format.open_memmap(f"{OUT}/real_img.npy", "w+", np.uint8, (n, H, W))
    msk = np.lib.format.open_memmap(f"{OUT}/real_msk.npy", "w+", np.uint8, (n, H, W))
    meta = []
    for i, (ds, stem, im, m, _) in enumerate(res):
        img[i], msk[i] = im, m
        meta.append({"i": i, "dataset": ds, "id": stem})
    img.flush(); msk.flush()

    rng = np.random.default_rng(0)
    fractions = {f: [] for f in pd.FRACTIONS}
    for ds in pd.LOADERS:
        idx = np.array([m["i"] for m in meta if m["dataset"] == ds])
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = int(round(0.70 * len(idx))), int(round(0.15 * len(idx)))
        for i in idx[:n_tr]:
            meta[i]["split"] = "train"
        for i in idx[n_tr:n_tr + n_va]:
            meta[i]["split"] = "val"
        for i in idx[n_tr + n_va:]:
            meta[i]["split"] = "test"
        for f in pd.FRACTIONS:
            fractions[f] += [int(i) for i in idx[:n_tr][:max(1, math.ceil(f / 100 * n_tr))]]
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    json.dump({str(k): sorted(v) for k, v in fractions.items()}, open(f"{OUT}/fractions.json", "w"))

    old = json.load(open(f"{OLD}/meta.json"))
    same = len(old) == len(meta) and all(a["dataset"] == b["dataset"] and a["id"] == b["id"]
                                         and a["split"] == b["split"] for a, b in zip(old, meta))
    print("image order and splits identical to the uncropped build:", same, flush=True)
    print("splits:", {s: sum(m["split"] == s for m in meta) for s in ("train", "val", "test")}, flush=True)

    for f in ("syn_img.npy", "syn_names.json"):
        lnk = f"{OUT}/{f}"
        if not os.path.exists(lnk):
            os.symlink(f"{OLD}/{f}", lnk)
    print("done", flush=True)


if __name__ == "__main__":
    main()
