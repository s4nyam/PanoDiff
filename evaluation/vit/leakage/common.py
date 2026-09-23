"""Shared pieces for the ViT leakage check: cached images, duplicate groups, the two splits."""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, WORK + "/temp-codes")
from vit_attention import ViT  # noqa: E402,F401  (the architecture of the paper's classifier)

OUT = WORK + "/vit-leakage/out"


def load():
    g = json.load(open(f"{OUT}/groups.json"))
    real = np.load(f"{OUT}/cache_real.npy", mmap_mode="r")
    syn = np.load(f"{OUT}/cache_syn.npy", mmap_mode="r")
    return real, syn, np.array(g["group_of_file"]), g


def random_split(n, seed, val_frac=0.2):
    """build_split of vit_attention.py: one permutation, the same indices for both classes."""
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:], idx[:cut], idx[cut:]


def grouped_split(groups, seed, val_frac=0.2):
    """Every copy of a radiograph on the same side. Whole groups go to the held-out side, in random
    order, until it holds as many real files as the random split does; the synthetic files are split
    at random into the same two sizes, so both classes stay balanced on each side."""
    n = len(groups)
    target = n - int(n * (1 - val_frac))
    rng = np.random.default_rng(seed)
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(int(g), []).append(i)
    order = list(members); rng.shuffle(order)
    va = []
    for g in order:
        if len(va) >= target:
            break
        va += members[g]
    va = np.array(sorted(va)); tr = np.setdiff1d(np.arange(n), va)
    sp = np.random.default_rng(seed + 1000).permutation(n)
    return tr, va, sp[:len(tr)], sp[len(tr):len(tr) + len(va)]


def leaked(groups, tr_real, va_real):
    """Held-out real files whose radiograph also appears among the training real files."""
    seen = set(groups[tr_real].tolist())
    return np.array([groups[i] in seen for i in va_real])


class Arr(Dataset):
    """(image, label) pairs from the cache, scaled exactly as vit_attention.Pairs does."""
    def __init__(self, real, syn, ir, isy):
        self.items = [(real, int(i), 0) for i in ir] + [(syn, int(i), 1) for i in isy]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, k):
        a, i, y = self.items[k]
        x = torch.from_numpy(np.asarray(a[i], np.float32) / 255.0)[None]
        return x * 2 - 1, y


@torch.no_grad()
def predict(model, arr, idx, dev, bs=64):
    out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.stack([np.asarray(arr[i], np.float32) for i in idx[s:s + bs]]) / 255.0)[:, None]
        logits, _ = model((x * 2 - 1).to(dev))
        out.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], int)
