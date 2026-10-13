"""Precision and recall of the Table 10 models on repeated random subsets of the generated images
(Section 4.5.2).

Replaces recall_subsample.py of the earlier versions of this repository, which gave the recall of
three models: the same ten random subsets of 5000 generated images per model (same generator seed,
so its recall values are reproduced up to floating-point differences between devices), now for
every model of Table 10 and for precision as well. k = 3 and the real reference set (all 7243 real
images, set 10) are as in pr_ratios_t11.py.

  precision of a subset = share of its images that lie within the neighbourhood of a real image
  recall of a subset    = share of the 7243 real images that lie within the neighbourhood of one of its images

Runs on CPU or GPU. Writes out/pr_subsample.json (per model: the ten values, mean, SD with n - 1,
and a 95% confidence interval of the mean from the t distribution with 9 degrees of freedom).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, numpy as np, torch
E = WORK + "/experiments/sr-distribution-metrics/out"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); K = 3; RUNS = 10; N = 5000
T975 = 2.2621571627409915                      # t quantile 0.975, 9 degrees of freedom
ARMS = ["4", "5", "sg2ada", "sg3", "fastgan_lite64", "adm", "ldm", "panodiff_hr"]
f = lambda n: torch.from_numpy(np.load(f"{E}/feats/{n}.npz")["pool"]).float().to(dev)
R = f("10")
def radii(X):
    return torch.cat([torch.cdist(X[i:i+1024], X).kthvalue(K + 1, dim=1).values for i in range(0, len(X), 1024)])
rR = radii(R)
def recall(G):
    rG = radii(G)
    s = torch.cat([(torch.cdist(R[i:i+1024], G) / rG[None]).min(1).values for i in range(0, len(R), 1024)])
    return float((s <= 1).float().mean())
def precision(G):
    s = torch.cat([(torch.cdist(G[i:i+1024], R) / rR[None]).min(1).values for i in range(0, len(G), 1024)])
    return float((s <= 1).float().mean())
def summary(v):
    v = np.array(v, float); m = v.mean(); sd = v.std(ddof=1); h = T975 * sd / np.sqrt(len(v))
    return dict(mean=float(m), sd=float(sd), ci95=[float(m - h), float(m + h)], min=float(v.min()), max=float(v.max()), runs=[float(x) for x in v])
out = {"settings": dict(k=K, runs=RUNS, subset=N, seed=0, device=str(dev), real_set="10 (7243 images)")}
for arm in ARMS:
    G = f(arm); g = torch.Generator().manual_seed(0); pr, rc = [], []
    for _ in range(RUNS):
        idx = torch.randperm(len(G), generator=g)[:N].to(dev)
        rc.append(recall(G[idx])); pr.append(precision(G[idx]))
    out[arm] = dict(n_generated=int(len(G)), precision=summary(pr), recall=summary(rc))
    print(arm, f"precision {100*np.mean(pr):.2f} +- {100*np.std(pr, ddof=1):.2f}   recall {100*np.mean(rc):.2f} +- {100*np.std(rc, ddof=1):.2f}", flush=True)
json.dump(out, open(f"{E}/pr_subsample.json", "w"), indent=2)
print("wrote pr_subsample.json")
