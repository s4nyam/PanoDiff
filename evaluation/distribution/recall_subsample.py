"""Robustness of the recall ranking in Table 11 to the draw of generated images: recall (k = 3,
as in pr_ratios_t11.py) recomputed on 10 random subsets of 5000 generated images per arm,
against all 7243 real images. Writes out/recall_subsample.json."""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, numpy as np, torch
E = WORK + "/swinir-ft-eval/out"
dev = torch.device("cuda"); K = 3
f = lambda n: torch.from_numpy(np.load(f"{E}/feats/{n}.npz")["pool"]).float().to(dev)
R = f("10")
def radii(X):
    return torch.cat([torch.cdist(X[i:i+1024], X).kthvalue(K + 1, dim=1).values for i in range(0, len(X), 1024)])
def recall(G):
    rG = radii(G)
    s = torch.cat([(torch.cdist(R[i:i+1024], G) / rG[None]).min(1).values for i in range(0, len(R), 1024)])
    return float((s <= 1).float().mean())
out = {}
for arm in ["4", "sg3", "sg2ada"]:
    G = f(arm); g = torch.Generator().manual_seed(0); v = []
    for _ in range(10):
        idx = torch.randperm(len(G), generator=g)[:5000].to(dev)
        v.append(recall(G[idx]))
    out[arm] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)), runs=v)
    print(arm, f"recall on 5000-image subsets: {np.mean(v):.4f} +- {np.std(v):.4f} (min {np.min(v):.4f}, max {np.max(v):.4f})", flush=True)
json.dump(out, open(f"{E}/recall_subsample.json", "w"), indent=2)
