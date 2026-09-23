"""Per-image precision/recall scores (as in pr_ratios.py / Figure 17) for every model of Table 11.

Same features (Inception pool, torchmetrics network), same k = 3 and the same ratio definition:
    fidelity score of generated g = min over real r      of |g - r| / radius_real(r)
    coverage score of real r      = min over generated g of |r - g| / radius_gen(g)
share <= 1 = precision / recall. Rows 1 and 2 of Table 11 (sets 4 and 5) are checked against
Table 9; FID is recomputed from the same features as a check against Table 11.

Writes out/pr_ratios_t11.npz and out/pr_t11.json.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json
import numpy as np
from scipy import linalg

E = WORK + "/swinir-ft-eval/out"
ARMS = ["4", "5", "sg2ada", "sg3", "fastgan_lite64", "adm", "ldm", "panodiff_hr"]
K = 3


def feats(n):
    return np.load(f"{E}/feats/{n}.npz")["pool"].astype(np.float32)


def sqdist(A, B):
    a2 = (A * A).sum(1)[:, None]; b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


def radii(X):
    out = []
    for i in range(0, len(X), 2048):
        d = np.sqrt(sqdist(X[i:i + 2048], X))
        out.append(np.partition(d, K, axis=1)[:, K])          # k-th neighbour, self at index 0
    return np.concatenate(out)


def scores(Q, X, rX):
    out = []
    for i in range(0, len(Q), 2048):
        out.append((np.sqrt(sqdist(Q[i:i + 2048], X)) / rX[None, :]).min(1))
    return np.concatenate(out)


def fid(A, B):
    A = A.astype(np.float64); B = B.astype(np.float64)
    m1, m2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs.real))


if __name__ == "__main__":
    ref = json.load(open(f"{E}/metrics_451.json"))["vs_real"]
    R = feats("10"); rR = radii(R)
    res, summary = {}, {}
    for a in ARMS:
        G = feats(a); rG = radii(G)
        f_, c_ = scores(G, R, rR), scores(R, G, rG)
        p, r = float((f_ <= 1).mean()), float((c_ <= 1).mean())
        F = fid(G, R)
        summary[a] = dict(n=int(len(G)), precision=p, recall=r, fid=F,
                          fid_median_ratio=float(np.median(f_)), cov_median_ratio=float(np.median(c_)))
        chk = ""
        if f"10-{a}" in ref:
            chk = f"  (Table 9: P {ref[f'10-{a}']['precision']:.4f} R {ref[f'10-{a}']['recall']:.4f} FID {ref[f'10-{a}']['fid']:.2f})"
        print(f"{a:15s} n={len(G)}  P {p:.4f}  R {r:.4f}  FID {F:.2f}{chk}", flush=True)
        res[f"{a}_fid"], res[f"{a}_cov"] = f_, c_
    np.savez(f"{E}/pr_ratios_t11.npz", **res)
    json.dump(summary, open(f"{E}/pr_t11.json", "w"), indent=2)
    print("wrote pr_ratios_t11.npz, pr_t11.json")
