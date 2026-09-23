"""Build the downstream tooth-segmentation data from the four source datasets that carry
pixel-level tooth outlines, plus the synthetic PanoDiff-SR set.

Task: binary tooth segmentation (tooth vs. everything else) -- the one task all four
datasets support. USPFORP is excluded: its annotations are 4-point boxes, not outlines.

  ADLD    labelme polygons, one per tooth, FDI numbers 11-48 (non-tooth labels dropped)
  TSXK    colour-coded semantic masks (masks_machine): tooth = any non-black pixel
  TUFTS   binary teeth masks (Segmentation/teeth_mask)
  DENTEX  COCO polygons from the quadrant-enumeration subset

Everything is resized to 512x1024 (H x W): images with INTER_AREA, masks with
INTER_NEAREST. Splits are 70/15/15 per dataset (seed 0), so every split has the same
dataset mix. Label fractions 5/10/25/50/100% are NESTED subsets of the training split,
also stratified by dataset, so a smaller fraction is always contained in a larger one.

Outputs in data/: real_img.npy, real_msk.npy (uint8, N x 512 x 1024), syn_img.npy,
meta.json (per-image dataset/id/split), fractions.json (train indices per fraction).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import glob, json, math, os, re
from multiprocessing import Pool
import numpy as np
import cv2

O = WORK + "/project-files/PanoDiff/GT-PanoDiff-Training/Original_Datasets"
SYN = WORK + "/project-files/PanoDiff/Syn-Calc-FID&IS/4-DiffHATSR"
OUT = WORK + "/downstream-seg/data"
H, W = 512, 1024
FRACTIONS = [5, 10, 25, 50, 100]
FDI = re.compile(r"^[1-4][1-8]$")


def finish(img, msk):
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.uint8), (msk > 0).astype(np.uint8)


def fill(shape, polygons):
    m = np.zeros(shape, np.uint8)
    for p in polygons:
        pts = np.round(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def items_adld():
    base = f"{O}/ADLD - A dual-labeled dataset"
    for j in sorted(glob.glob(f"{base}/labels/*.json")):
        stem = os.path.splitext(os.path.basename(j))[0]
        img = next(iter(glob.glob(f"{base}/images/{stem}.*")), None)
        if img:
            yield ("ADLD", stem, img, j)


def load_adld(img_p, js):
    img = cv2.imread(img_p, 0)
    d = json.load(open(js))
    polys = [s["points"] for s in d["shapes"]
             if s.get("shape_type", "polygon") == "polygon" and FDI.match(str(s["label"]))]
    return finish(img, fill(img.shape, polys))


def items_tsxk():
    base = f"{O}/TSXK - Teeth Segmentation on dental X-ray images Kaggle/Teeth Segmentation PNG/d2"
    for m in sorted(glob.glob(f"{base}/masks_machine/*.png")):
        stem = os.path.splitext(os.path.basename(m))[0]
        img = next(iter(glob.glob(f"{base}/img/{stem}.*")), None)
        if img:
            yield ("TSXK", stem, img, m)


def load_tsxk(img_p, m_p):
    img = cv2.imread(img_p, 0)
    m = cv2.imread(m_p, cv2.IMREAD_UNCHANGED)
    m = (m.reshape(m.shape[0], m.shape[1], -1).max(axis=2) > 0).astype(np.uint8)
    if m.shape != img.shape:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return finish(img, m)


def items_tufts():
    base = f"{O}/TUFTS - Tufts Panoramic Dataset/Tufts Database"
    for m in sorted(glob.glob(f"{base}/Segmentation/teeth_mask/*")):
        stem = os.path.splitext(os.path.basename(m))[0]
        img = next(iter(glob.glob(f"{base}/Radiographs/{stem}.*")), None)
        if img:
            yield ("TUFTS", stem, img, m)


def load_tufts(img_p, m_p):
    img = cv2.imread(img_p, 0)
    m = (cv2.imread(m_p, 0) > 127).astype(np.uint8)
    if m.shape != img.shape:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return finish(img, m)


def items_dentex():
    base = f"{O}/DENTEX - Panoramic Dataset/training_data/training_data/quadrant_enumeration"
    d = json.load(open(f"{base}/train_quadrant_enumeration.json"))
    polys = {}
    for a in d["annotations"]:
        polys.setdefault(a["image_id"], []).extend(a["segmentation"])
    for im in d["images"]:
        p = f"{base}/xrays/{im['file_name']}"
        if os.path.exists(p) and im["id"] in polys:
            yield ("DENTEX", os.path.splitext(im["file_name"])[0], p, polys[im["id"]])


def load_dentex(img_p, polys):
    img = cv2.imread(img_p, 0)
    return finish(img, fill(img.shape, polys))


LOADERS = {"ADLD": load_adld, "TSXK": load_tsxk, "TUFTS": load_tufts, "DENTEX": load_dentex}


def work(item):
    ds, stem, img_p, ann = item
    try:
        img, msk = LOADERS[ds](img_p, ann)
        return ds, stem, img, msk, None
    except Exception as e:
        return ds, stem, None, None, f"{type(e).__name__}: {e}"


def work_syn(p):
    im = cv2.imread(p, 0)
    if im.shape != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    items = list(items_adld()) + list(items_tsxk()) + list(items_tufts()) + list(items_dentex())
    print("real items:", len(items), {d: sum(i[0] == d for i in items) for d in LOADERS}, flush=True)
    with Pool(os.cpu_count()) as pool:
        res = pool.map(work, items, chunksize=8)
    bad = [(r[0], r[1], r[4]) for r in res if r[4]]
    empty = [(r[0], r[1]) for r in res if r[4] is None and r[3].sum() == 0]
    if bad:
        print("FAILED to load:", len(bad), bad[:5], flush=True)
    if empty:
        print("dropped (no tooth pixels):", len(empty), empty[:5], flush=True)
    res = [r for r in res if r[4] is None and r[3].sum() > 0]

    n = len(res)
    img = np.lib.format.open_memmap(f"{OUT}/real_img.npy", "w+", np.uint8, (n, H, W))
    msk = np.lib.format.open_memmap(f"{OUT}/real_msk.npy", "w+", np.uint8, (n, H, W))
    meta = []
    for i, (ds, stem, im, m, _) in enumerate(res):
        img[i], msk[i] = im, m
        meta.append({"i": i, "dataset": ds, "id": stem})
    img.flush(); msk.flush()

    rng = np.random.default_rng(0)
    fractions = {f: [] for f in FRACTIONS}
    for ds in LOADERS:
        idx = np.array([m["i"] for m in meta if m["dataset"] == ds])
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = int(round(0.70 * len(idx))), int(round(0.15 * len(idx)))
        for i in idx[:n_tr]:
            meta[i]["split"] = "train"
        for i in idx[n_tr:n_tr + n_va]:
            meta[i]["split"] = "val"
        for i in idx[n_tr + n_va:]:
            meta[i]["split"] = "test"
        train = idx[:n_tr]
        for f in FRACTIONS:
            fractions[f] += [int(i) for i in train[:max(1, math.ceil(f / 100 * len(train)))]]
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    json.dump({str(k): sorted(v) for k, v in fractions.items()}, open(f"{OUT}/fractions.json", "w"))
    counts = {s: sum(m["split"] == s for m in meta) for s in ("train", "val", "test")}
    print("splits:", counts, "| fractions:", {f: len(v) for f, v in fractions.items()}, flush=True)
    print("tooth-pixel fraction per dataset:",
          {ds: round(float(np.mean([msk[m["i"]].mean() for m in meta if m["dataset"] == ds])), 4)
           for ds in LOADERS}, flush=True)

    syn = sorted(glob.glob(f"{SYN}/*.png"))
    print("synthetic images:", len(syn), flush=True)
    sim = np.lib.format.open_memmap(f"{OUT}/syn_img.npy", "w+", np.uint8, (len(syn), H, W))
    with Pool(os.cpu_count()) as pool:
        for i, im in enumerate(pool.imap(work_syn, syn, chunksize=16)):
            sim[i] = im
    sim.flush()
    json.dump([os.path.basename(p) for p in syn], open(f"{OUT}/syn_names.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
