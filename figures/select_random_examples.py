"""Draw the examples of the direct high-resolution comparison figure (Figure 19), the same way
for every row: three images at random, none chosen by eye.

For every row (the real radiographs, PanoDiff-SR and the five direct high-resolution models of
the anatomical table) the file names of the image set are sorted and permuted with numpy's
default_rng(0), and the first three images of that permutation are taken. This is the
permutation with which anatomy.py draws the 1500 images per set behind the anatomical measures
(Table 11), so the examples are the first three images of that sample: the order was fixed
before the figure was drawn, and the measures of the three images are already on file.

Nothing is excluded and nothing is replaced. The picks, with their three measures and whether
each lies inside the central 95% range of the real radiographs (the outlier rule of Table 11),
are written to random_examples.json, which make_mosaics.py reads.

With --check the measures of every pick are computed again with anatomy.measure() and compared
with the stored ones; this confirms that the stored order is reproduced (needs scipy).

Replaces select_best_examples.py of earlier versions of this repository (rows ranked by an
anatomical criterion that used the crown count, and a PanoDiff-SR row of radiographs the observers
judged real).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import json, os, sys
import numpy as np

A = WORK + "/experiments/direct-hr-baselines"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_examples.json")
# row key = the name anatomy.py gives the set (out/anat_<key>.json)
ROWS = [("real", "refs/real"), ("PanoDiffSR_existing", "refs/PanoDiffSR_existing"),
        ("sg2ada", "samples/sg2ada"), ("sg3", "samples/sg3"), ("adm", "samples/adm"),
        ("ldm", "samples/ldm"), ("panodiff_hr", "samples/panodiff_hr")]
MEAS = ["symmetry", "periodicity", "occlusal"]      # the three measures of Table 11
NCOL, SEED, CENTRAL = 3, 0, 95


def drawn(rel):
    """the first NCOL files of the set, in the order of anatomy.py"""
    folder = os.path.join(A, rel)
    fs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rng = np.random.default_rng(SEED)                  # same draw as anatomy.py
    return len(fs), [os.path.join(folder, fs[i]) for i in rng.permutation(len(fs))[:NCOL]]


def main():
    check = "--check" in sys.argv
    if check:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "anatomy_memorisation"))
        import anatomy
    real = json.load(open(f"{A}/analysis_r25_r31/out/anat_real.json"))
    tail = (100 - CENTRAL) / 2
    lo = {m: float(np.percentile(real[m], tail)) for m in MEAS}
    hi = {m: float(np.percentile(real[m], 100 - tail)) for m in MEAS}

    result = {"rule": f"first {NCOL} files of numpy default_rng({SEED}).permutation over the sorted file "
                      "names of each set (the order of the 1500-image sample of anatomy.py); "
                      "nothing excluded, nothing replaced",
              "central_range_percent": CENTRAL, "real_range": {m: [lo[m], hi[m]] for m in MEAS},
              "picks": {}, "rows": {}}
    for key, rel in ROWS:
        n, paths = drawn(rel)
        d = json.load(open(f"{A}/analysis_r25_r31/out/anat_{key}.json"))
        result["picks"][key] = paths
        result["rows"][key] = []
        for k, p in enumerate(paths):
            vals = {m: float(d[m][k]) for m in MEAS}
            if check:
                sym, per, _, occ = anatomy.measure(p)
                if not np.allclose([sym, per, occ], [vals[m] for m in MEAS], atol=1e-6):
                    sys.exit(f"{key}: pick {k} ({os.path.basename(p)}) does not reproduce the stored measures")
            inside = all(lo[m] <= vals[m] <= hi[m] for m in MEAS)
            result["rows"][key].append({"file": os.path.basename(p), **{m: round(vals[m], 3) for m in MEAS},
                                        "inside_real_range": inside})
        print(f"{key:20s} {n} files  " + "  ".join(
            f"{r['file']} ({'in' if r['inside_real_range'] else 'OUT'})" for r in result["rows"][key]))
    if check:
        print("stored measures reproduced for every pick")
    json.dump(result, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
