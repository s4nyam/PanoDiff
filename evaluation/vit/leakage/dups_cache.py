"""Step 0 of the ViT leakage check: duplicate groups over the 7243 real files, and an image cache.

Duplicate groups: two real files are the same radiograph if their bytes are identical (md5) or if
their normalised thumbnails agree to >= 0.98 at 64x32 and again at 256x128 (the rule that found the
115 DENTEX copies). Groups are the connected components of that relation.

Cache: every real (10-TrainDataInHR) and synthetic (4-DiffHATSR) file exactly as the ViT sees it
(vit_attention.py Pairs: grey, BILINEAR to 512x256, uint8), in the sorted order build_split uses,
so training and evaluation read one array instead of 14486 PNGs.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import hashlib, json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image

sys.path.insert(0, WORK + "/temp-codes")
from vit_attention import REAL_DIR, SYNTH_DIR, H, W  # noqa: E402

OUT = WORK + "/vit-leakage/out"
EXT = (".png", ".jpg", ".jpeg")


def lst(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXT))


def norm(a):
    a = a.astype(np.float32).ravel(); a -= a.mean(); n = np.linalg.norm(a)
    return a / (n if n > 0 else 1)


def real_item(p):
    raw = open(p, "rb").read()
    im = Image.open(p).convert("L")
    vit = np.asarray(im.resize((W, H), Image.BILINEAR), np.uint8)
    return (hashlib.md5(raw).hexdigest(), vit, norm(np.asarray(im.resize((64, 32), Image.BILINEAR))),
            norm(np.asarray(im.resize((256, 128), Image.BILINEAR))))


def syn_item(p):
    return np.asarray(Image.open(p).convert("L").resize((W, H), Image.BILINEAR), np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    real, syn = lst(REAL_DIR), lst(SYNTH_DIR)
    n = min(len(real), len(syn)); real, syn = real[:n], syn[:n]
    with Pool(os.cpu_count()) as pool:
        R = pool.map(real_item, real, chunksize=16)
        S = pool.map(syn_item, syn, chunksize=16)
    np.save(f"{OUT}/cache_real.npy", np.stack([r[1] for r in R]))
    np.save(f"{OUT}/cache_syn.npy", np.stack(S))
    md5 = [r[0] for r in R]; t64 = np.stack([r[2] for r in R]); t256 = np.stack([r[3] for r in R])

    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b: parent[max(a, b)] = min(a, b)
    first = {}
    for i, h in enumerate(md5):
        if h in first: union(first[h], i)
        else: first[h] = i
    n_exact = sum(1 for i in range(n) if find(i) != i)
    near = 0
    for s in range(0, n, 512):
        C = t64[s:s + 512] @ t64.T
        for r, i in enumerate(range(s, min(s + 512, n))):
            for j in np.nonzero(C[r] >= 0.98)[0]:
                if j > i and float(t256[i] @ t256[j]) >= 0.98 and find(i) != find(j):
                    union(i, j); near += 1
    roots = [find(i) for i in range(n)]
    groups = {}
    for i, g in enumerate(roots):
        groups.setdefault(g, []).append(i)
    sizes = np.bincount([len(v) for v in groups.values()])
    res = {"n_files": n, "n_distinct": len(groups), "n_repeated_files": n - len(groups),
           "exact_md5_copies": n_exact, "extra_near_identical_links": near,
           "group_size_histogram": {int(k): int(v) for k, v in enumerate(sizes) if v},
           "group_of_file": roots, "real_files": [os.path.basename(p) for p in real],
           "syn_files": [os.path.basename(p) for p in syn]}
    json.dump(res, open(f"{OUT}/groups.json", "w"))
    print({k: v for k, v in res.items() if k not in ("group_of_file", "real_files", "syn_files")}, flush=True)


if __name__ == "__main__":
    main()
