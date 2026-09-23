"""FID, KID, IS and precision/recall from the cached Inception features (extract_feats.py).

The formulas are torchmetrics' own, so FID and IS reproduce the paper's fid.py / is.py:
  FID  mean and covariance accumulated in float64 (as FrechetInceptionDistance.update does),
       then torchmetrics' _compute_fid.
  KID  torchmetrics KernelInceptionDistance defaults: 100 subsets of 1000, polynomial kernel of
       degree 3, gamma = 1/2048, coef 1; reported as mean and sd, x 10^3. Seeded.
  IS   torchmetrics InceptionScore: softmax of 'logits_unbiased', random permutation, 10 splits,
       mean and sd of exp(KL). Seeded (the paper's run was not, so IS agrees to ~1e-3).
  P/R  improved precision and recall (Kynkaanniemi et al., NeurIPS 2019) on the same 2048-d
       features, k = 3: precision = share of generated samples inside the real manifold,
       recall = share of real samples inside the generated manifold.
Protocol checks against published values: FID(10,6) = 91.10, FID(10,4) = 40.5, FID(1,2) = 55.0,
IS(6) = 1.557, IS(4) = 2.381.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, sys
import numpy as np
import torch
from torchmetrics.image.fid import _compute_fid
from torchmetrics.image.kid import poly_mmd

FEATS = WORK + "/swinir-ft-eval/out/feats"
_cache = {}


def load(n):
    if n not in _cache:
        z = np.load(f"{FEATS}/{n}.npz")
        _cache[n] = (torch.from_numpy(z["pool"]), torch.from_numpy(z["logits"]))
    return _cache[n]


def fid(a, b):
    def stats(f):
        f = f.double(); n = f.shape[0]
        mu = (f.sum(0) / n).unsqueeze(0)
        cov = (f.t().mm(f) - n * mu.t().mm(mu)) / (n - 1)
        return mu.squeeze(0), cov
    m1, s1 = stats(load(a)[0]); m2, s2 = stats(load(b)[0])
    return float(_compute_fid(m1, s1, m2, s2))


def kid(a, b, subsets=100, size=1000):
    torch.manual_seed(0)
    r, f = load(a)[0].double(), load(b)[0].double()
    s = []
    for _ in range(subsets):
        pr = torch.randperm(r.shape[0]); pf = torch.randperm(f.shape[0])
        s.append(poly_mmd(r[pr[:size]], f[pf[:size]], degree=3, gamma=None, coef=1.0))
    s = torch.stack(s)
    return float(s.mean()) * 1e3, float(s.std(unbiased=False)) * 1e3


def inception_score(a, splits=10):
    torch.manual_seed(0)
    x = load(a)[1].double()
    x = x[torch.randperm(x.shape[0])]
    p, lp = x.softmax(1), x.log_softmax(1)
    kl = []
    for pc, lpc in zip(p.chunk(splits), lp.chunk(splits)):
        m = pc.mean(0, keepdim=True)
        kl.append((pc * (lpc - m.log())).sum(1).mean().exp())
    kl = torch.stack(kl)
    return float(kl.mean()), float(kl.std())


def prec_rec(a, b, k=3):
    """a = real, b = generated. Radii: distance to the k-th nearest neighbour within the same set."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, G = load(a)[0].to(dev), load(b)[0].to(dev)
    def radii(X):
        out = []
        for i in range(0, X.shape[0], 1024):
            d = torch.cdist(X[i:i + 1024], X)
            out.append(d.kthvalue(k + 1, dim=1).values)   # k+1: the point itself is at distance 0
        return torch.cat(out)
    rR, rG = radii(R), radii(G)
    def covered(Q, X, rX):
        hit = []
        for i in range(0, Q.shape[0], 1024):
            hit.append((torch.cdist(Q[i:i + 1024], X) <= rX.unsqueeze(0)).any(1))
        return float(torch.cat(hit).float().mean())
    return covered(G, R, rR), covered(R, G, rG)


VS_REAL = [("1", "2"), ("1", "3"), ("10", "4"), ("10", "5"), ("10", "6ft"), ("10", "7ft"),
           ("10", "8"), ("10", "9ft")]
BETWEEN = [("2", "3"), ("4", "5"), ("4", "6ft"), ("4", "7ft"), ("5", "6ft"), ("5", "7ft"), ("6ft", "7ft")]
IS_SETS = ["1", "2", "3", "4", "5", "6ft", "7ft", "8", "9ft", "10"]

if __name__ == "__main__":
    out = {"check": {}, "vs_real": {}, "between": {}, "is": {}}
    out["check"]["fid_10_6 (paper 91.10)"] = fid("10", "6")
    out["check"]["fid_10_4 (paper 40.5)"] = fid("10", "4")
    out["check"]["fid_1_2 (paper 55.0)"] = fid("1", "2")
    out["check"]["is_6 (paper 1.557)"] = inception_score("6")[0]
    out["check"]["is_4 (paper 2.381)"] = inception_score("4")[0]
    print("checks:", json.dumps(out["check"], indent=1), flush=True)
    for a, b in VS_REAL:
        km, ks = kid(a, b); p, r = prec_rec(a, b)
        out["vs_real"][f"{a}-{b}"] = dict(fid=fid(a, b), kid=km, kid_sd=ks, precision=p, recall=r)
        print(f"{a:>3}-{b:<4} FID {out['vs_real'][f'{a}-{b}']['fid']:7.2f}  KID {km:6.2f}+-{ks:4.2f}  P {p:.3f}  R {r:.3f}", flush=True)
    for a, b in BETWEEN:
        km, ks = kid(a, b)
        out["between"][f"{a}-{b}"] = dict(fid=fid(a, b), kid=km, kid_sd=ks)
        print(f"{a:>3}-{b:<4} FID {out['between'][f'{a}-{b}']['fid']:7.2f}  KID {km:6.2f}+-{ks:4.2f}", flush=True)
    for s in IS_SETS:
        m, sd = inception_score(s)
        out["is"][s] = dict(mean=m, sd=sd)
        print(f"IS {s:>4}: {m:.3f} +- {sd:.3f}", flush=True)
    json.dump(out, open(FEATS + "/../metrics_451.json", "w"), indent=1)
