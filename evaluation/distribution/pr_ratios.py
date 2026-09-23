"""Per-image scores behind the precision and recall of Table 9, for the figure that shows them.

Improved precision/recall (Kynkaanniemi et al., 2019, k = 3) asks, per image, whether it lies
inside the neighbourhood of at least one image of the other set, a neighbourhood being the ball
out to that image's k-th nearest neighbour within its own set. Written as a ratio,
    fidelity score of generated g = min over real r      of |g - r| / radius_real(r)
    coverage score of real r      = min over generated g of |r - g| / radius_gen(g)
an image is inside when its score is <= 1, so the share of scores <= 1 is exactly the precision
(fidelity) or recall (coverage). Same features, same k and same float32 GPU arithmetic as
prec_rec() in metrics_from_feats.py; the script checks that it reproduces those numbers.

Writes out/pr_ratios.npz: <pair>_fid and <pair>_cov arrays, one value per image.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json
import numpy as np
import torch

E = WORK + "/swinir-ft-eval/out"
PAIRS = [("1", "2"), ("1", "3"), ("10", "4"), ("10", "5"), ("10", "6ft"), ("10", "7ft")]
K = 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feats(n):
    return torch.from_numpy(np.load(f"{E}/feats/{n}.npz")["pool"]).to(dev)


def radii(X):
    out = []
    for i in range(0, X.shape[0], 1024):
        out.append(torch.cdist(X[i:i + 1024], X).kthvalue(K + 1, dim=1).values)  # k+1: self at 0
    return torch.cat(out)


def scores(Q, X, rX):
    """For every row of Q: min over X of |q - x| / radius(x)."""
    out = []
    for i in range(0, Q.shape[0], 1024):
        out.append((torch.cdist(Q[i:i + 1024], X) / rX.unsqueeze(0)).min(1).values)
    return torch.cat(out).cpu().numpy()


if __name__ == "__main__":
    ref = json.load(open(f"{E}/metrics_451.json"))["vs_real"]
    res, real_cache = {}, {}
    for a, b in PAIRS:
        if a not in real_cache:
            R = feats(a); real_cache[a] = (R, radii(R))
        R, rR = real_cache[a]
        G = feats(b); rG = radii(G)
        fid, cov = scores(G, R, rR), scores(R, G, rG)
        p, r = float((fid <= 1).mean()), float((cov <= 1).mean())
        P, Rr = ref[f"{a}-{b}"]["precision"], ref[f"{a}-{b}"]["recall"]
        print(f"{a}-{b}: precision {p:.4f} (table {P:.4f})  recall {r:.4f} (table {Rr:.4f})", flush=True)
        assert abs(p - P) < 5e-4 and abs(r - Rr) < 5e-4, "does not reproduce Table 9"
        res[f"{a}-{b}_fid"], res[f"{a}-{b}_cov"] = fid, cov
    np.savez(f"{E}/pr_ratios.npz", **res)
    print("wrote", f"{E}/pr_ratios.npz")
