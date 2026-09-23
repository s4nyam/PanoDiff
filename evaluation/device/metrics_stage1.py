"""
Stage 1 of the metric recomputation: embed one image set with the Inception
network and cache everything needed for FID and IS.

Design note. FID only needs the mean and covariance of the 2048-d pool
features, so every set is embedded exactly once and the pairwise FIDs are then
computed analytically in stage 2. Embedding each of the ten sets once costs ten
forward passes instead of two per pair, and adding a new comparison later is
free.

Also cached:
  * statistics for two disjoint halves of the set, which give the split-half
    FID "floor" reported in the manuscript;
  * the unbiased logits, from which the Inception Score is computed.

The feature extractor and the FID estimator are torchmetrics' own, so the
numbers are directly comparable with the values in the project's
fid_scores.csv / inception_scores.csv, which were produced with torchmetrics.
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import NoTrainInceptionV3

EXT = (".png", ".jpg", ".jpeg")


class Folder(Dataset):
    def __init__(self, roots, limit=None, seed=0, resize=None, crop=None):
        if isinstance(roots, str):
            roots = [roots]
        names = []
        for r in roots:
            names += [os.path.join(r, f) for f in sorted(os.listdir(r))
                      if f.lower().endswith(EXT)]
        names = sorted(names)
        self.resize = resize
        self.crop = crop
        if limit is not None and len(names) > limit:
            # deterministic subsample so sets of unequal size can be compared
            rng = np.random.default_rng(seed)
            names = [names[i] for i in sorted(rng.choice(len(names), limit,
                                                         replace=False))]
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.names[i]).convert("RGB")
        if self.crop is not None:
            # the training corpus's fixed margins (top, left, bottom, right), capped at a
            # quarter of the side as in medsam-probe/code/prepare_data_crop.py
            t, l, b, r = self.crop
            w, h = img.size
            t, b = min(t, h // 4), min(b, h // 4)
            l, r = min(l, w // 4), min(r, w // 4)
            img = img.crop((l, t, w - r, h - b))
        if self.resize is not None:
            # match the manuscript's preprocessing: LANCZOS to the target size
            img = img.resize(self.resize, Image.LANCZOS)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        return x                       # uint8 CHW, as torchmetrics expects


def stats(feats):
    """mean and covariance in the same convention torchmetrics uses."""
    assert feats.ndim == 2, f"expected (N, D) features, got {tuple(feats.shape)}"
    n = feats.shape[0]
    s = feats.sum(0)
    cov_sum = feats.transpose(0, 1) @ feats
    mu = s / n
    cov = (cov_sum - n * torch.outer(mu, mu)) / (n - 1)
    return mu.cpu().numpy(), cov.cpu().numpy(), n


def inception_score(logits, splits=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(logits.shape[0], generator=g)
    logits = logits[idx]
    parts = []
    for chunk in logits.chunk(splits):
        p = chunk.softmax(dim=1)
        lp = chunk.log_softmax(dim=1)
        kl = p * (lp - p.mean(0, keepdim=True).log())
        parts.append(kl.sum(1).mean().exp())
    v = torch.stack(parts)
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="deterministically subsample to this many images")
    ap.add_argument("--resize", type=int, nargs=2, default=None,
                    metavar=("W", "H"), help="LANCZOS resize before embedding")
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("T", "L", "B", "R"),
                    help="crop these margins before the resize")
    ap.add_argument("--save-features", action="store_true",
                    help="also cache per-image features, so that matched "
                         "subsamples can be drawn later without re-embedding")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NoTrainInceptionV3(name="inception-v3-compat",
                             features_list=["2048", "logits_unbiased"]).to(dev)
    net.eval()

    ds = Folder(a.folder, a.limit,
                resize=tuple(a.resize) if a.resize else None,
                crop=tuple(a.crop) if a.crop else None)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=6, pin_memory=True)
    print(f"[{a.name}] {len(ds)} images from {a.folder}", flush=True)

    pool, logits = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(dl):
            # NoTrainInceptionV3.forward() returns only the first requested
            # feature; the underlying torch-fidelity forward returns the tuple
            # (pool2048, logits_unbiased) that both metrics need.
            f = net._torch_fidelity_forward(x.to(dev, non_blocking=True))
            pool.append(f[0].reshape(x.shape[0], -1).double().cpu())
            logits.append(f[1].reshape(x.shape[0], -1).double().cpu())
            if (i + 1) % 20 == 0:
                done = (i + 1) * a.batch
                print(f"[{a.name}] {done}/{len(ds)} "
                      f"({done/(time.time()-t0):.0f} img/s)", flush=True)
    pool = torch.cat(pool)
    logits = torch.cat(logits)

    mu, cov, n = stats(pool)
    # Split-half FID estimates the floor achievable at this sample size, so the
    # two halves must be exchangeable. Splitting in filename order is not:
    # the real training set is stored in dataset-concatenation order, so its
    # first and last halves come from different source devices and the
    # "floor" then measures cross-device distance instead (24.5 rather than
    # 6.9). A seeded random permutation makes the halves exchangeable.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    shuffled = pool[perm]
    half = n // 2
    mu_a, cov_a, na = stats(shuffled[:half])
    mu_b, cov_b, nb = stats(shuffled[half:2 * half])
    is_mean, is_std = inception_score(logits.float())

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    extra = {"features": pool.float().numpy()} if a.save_features else {}
    np.savez_compressed(a.out, name=a.name, folder=str(a.folder), n=n, **extra,
                        mu=mu, cov=cov,
                        mu_a=mu_a, cov_a=cov_a, n_a=na,
                        mu_b=mu_b, cov_b=cov_b, n_b=nb,
                        is_mean=is_mean, is_std=is_std)
    print(f"[{a.name}] DONE n={n}  IS={is_mean:.4f}+-{is_std:.4f}  "
          f"({time.time()-t0:.0f}s)  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
