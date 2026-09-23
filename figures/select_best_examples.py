"""Rank candidate 'best examples' for the direct-HR comparison figure, the same way for
every arm, from the anatomical measures already computed by anatomy.py.

Criterion (identical for every row):
  1. inside the central 98% range of REAL radiographs on all four measures, i.e. not a
     gross failure by the paper's own definition (Table 11, last row);
  2. no more crowns than the real images' 95th percentile, and occlusal depth within the
     real 10th-90th percentile: this excludes duplicated tooth rows (which inflate the
     crown count) and requires one clear bite line;
  3. ranked by z(n_crowns) + z(symmetry) + z(periodicity), each standardised against the
     real images: more distinct crowns rendered, a more symmetric arch, a more regular
     tooth row -- the visible signs of a complete, well-formed dentition;
  4. near-duplicates skipped (grayscale correlation > 0.9 with an earlier pick).
The top candidates are then inspected by eye; the final picks and this procedure are
recorded in the response document.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os
import numpy as np

A = WORK + "/new-baselines-generation"
OUT = WORK + "/new-files/figure-sources/best_examples.json"
ARMS = {"real": "refs/real", "sg2ada": "samples/sg2ada", "adm": "samples/adm",
        "ldm": "samples/ldm", "panodiff_hr": "samples/panodiff_hr"}
MEAS = ["symmetry", "periodicity", "n_crowns", "occlusal"]
TOP = 12


def files_for(rel):
    folder = os.path.join(A, rel)
    fs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rng = np.random.default_rng(0)                     # same draw as anatomy.py
    return [os.path.join(folder, fs[i]) for i in rng.permutation(len(fs))[:1500]]


real = json.load(open(f"{A}/analysis_r25_r31/out/anat_real.json"))
lo = {m: np.percentile(real[m], 1) for m in MEAS}
hi = {m: np.percentile(real[m], 99) for m in MEAS}
mu = {m: np.mean(real[m]) for m in MEAS}
sd = {m: np.std(real[m]) for m in MEAS}

result = {"criterion": __doc__, "arms": {}}
for arm, rel in ARMS.items():
    d = json.load(open(f"{A}/analysis_r25_r31/out/anat_{arm}.json"))
    X = {m: np.asarray(d[m], float) for m in MEAS}
    ok = np.all([(X[m] >= lo[m]) & (X[m] <= hi[m]) for m in MEAS], axis=0)
    ok &= X["n_crowns"] <= np.percentile(real["n_crowns"], 95)
    ok &= (X["occlusal"] >= np.percentile(real["occlusal"], 10)) & (X["occlusal"] <= np.percentile(real["occlusal"], 90))
    score = sum((X[m] - mu[m]) / sd[m] for m in ("n_crowns", "symmetry", "periodicity"))
    score[~ok] = -np.inf
    order = [int(i) for i in np.argsort(-score) if np.isfinite(score[i])][:TOP]
    paths = files_for(rel)
    result["arms"][arm] = [{"path": paths[i], "score": round(float(score[i]), 3),
                            **{m: round(float(X[m][i]), 3) for m in MEAS}} for i in order]
    print(f"{arm:12s} in-range {ok.sum():4d}/1500  top score {score[order[0]]:.2f}  "
          f"#{TOP} {score[order[-1]]:.2f}  crowns of top 3: {[result['arms'][arm][k]['n_crowns'] for k in range(3)]}")
json.dump(result, open(OUT, "w"), indent=2)
print("wrote", OUT)
