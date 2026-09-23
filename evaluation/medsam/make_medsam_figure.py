"""Figure for the MedSAM probing subsection: real and synthetic radiographs with the same
per-region box prompts and MedSAM's masks, twelve of each (three rows of four per set).

Selection (recorded in out/medsam_figure_picks.json, reported in the response document, not in
the caption): only radiographs whose dentition appears complete, judged on the U-Net reference
mask used for the prompts in the same way on both sets -- the tooth region spans at least 55%
of the image width, covers that span without gaps in both the upper and the lower arch
(coverage >= 0.98) and breaks into at least three regions -- and no letterboxed image. Among
those, images whose tooth region falls in the middle of the real set's size range (40th to
90th percentile, so that unusually close framings are left out) are ranked by the sum of
z-scored mean predicted IoU and mean Dice to the reference; the real rows take the top three of
each source device, the synthetic rows the top twelve.

Layout: methods as rows, four columns across the full text width (390 pt), each panel a 2:1
radiograph 97.5 x 48.75 pt with a 9 pt label strip above the row. Boxes are the prompts,
contours are MedSAM's masks, the badge is the mean predicted IoU over the panel's prompts.
Runs inside the LUMI PyTorch container (CPU is fine: eight image encodings).
"""
import csv, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from medsam_probe import load_medsam, embed, prompt, boxes_from_components, H, W, D, E   # noqa: E402

FIGS = f"{E}/new-files/overleaf/figures/fig20-medsam"
# The probe can be run on more than one preparation of the real set; PROBE_OUT/PROBE_DATA/
# PROBE_REF pick which one this figure is drawn from (default: the directories above).
POUT = os.environ.get("PROBE_OUT", f"{E}/medsam-probe/out")
PDATA = os.environ.get("PROBE_DATA", D)
PREF = os.environ.get("PROBE_REF", f"{E}/medsam-probe/data")
PT = 1 / 72.27
W_PT, NCOL, HEADER_PT = 390.0, 4, 9.0
PANEL_W, PANEL_H = W_PT / NCOL, W_PT / NCOL / 2
DEVICES = ["ADLD", "DENTEX", "TSXK", "TUFTS"]

rows = []
for p in sorted(glob.glob(f"{POUT}/probe_shard*.csv")):
    rows += [r for r in csv.DictReader(open(p)) if r["protocol"] == "teeth"]
per = {}
for r in rows:
    per.setdefault((r["set"], int(r["idx"]), r["source"]), []).append((float(r["pred_iou"]), float(r["dice_ref"])))
# Letterboxed radiographs (black bars top/bottom, a few ADLD files) are not shown.
_imgs = {"real": np.load(f"{PDATA}/real_img.npy", mmap_mode="r"), "syn": np.load(f"{PDATA}/syn_img.npy", mmap_mode="r")}
_refs = {"real": np.load(f"{PREF}/ref_real.npy", mmap_mode="r"),
         "syn": np.load(f"{PREF}/ref_syn.npy", mmap_mode="r")}
def letterboxed(which, i):
    a = _imgs[which][i]; band = H // 20
    return min(a[:band].mean(), a[-band:].mean()) < 12
def dentition(which, i):
    """(span fraction, arch coverage, area) of the reference tooth region: a complete dentition
    spans a wide arch and, teeth being adjacent, leaves no empty column in either arch."""
    ref = np.asarray(_refs[which][i]) > 0
    ys, xs = np.nonzero(ref)
    if len(xs) == 0:
        return 0.0, 0.0, 0
    x0, x1 = xs.min(), xs.max() + 1; ymid = int(np.median(ys))
    cov = min(ref[:ymid, x0:x1].any(0).mean(), ref[ymid:, x0:x1].any(0).mean())
    return (x1 - x0) / W, float(cov), int(len(xs))
dent = {k: dentition(k[0], k[1]) for k in per}
cand = [(k, np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in per.items()
        if len(v) >= 3 and not letterboxed(k[0], k[1]) and dent[k][0] >= 0.55 and dent[k][1] >= 0.98]
print("complete dentitions:", sum(k[0] == "real" for k, _, _ in cand), "real,", sum(k[0] == "syn" for k, _, _ in cand), "synthetic")
iou = np.array([c[1] for c in cand]); dsc = np.array([c[2] for c in cand])
# A full dentition fills more of the image than a gappy one, but the largest regions belong to
# radiographs framed unusually close, so the candidates are restricted to the middle of the real
# set's range of tooth-region area (40th to 90th percentile) before ranking. The window is read
# off the real images and then applied to both sets unchanged.
area = np.array([dent[c[0]][2] for c in cand], dtype=float)
lo, hi = np.quantile([dent[c[0]][2] for c in cand if c[0][0] == "real"], [0.30, 0.95])
inside = (area >= lo) & (area <= hi)
print(f"tooth-region area window {lo:.0f}-{hi:.0f} px ->",
      sum(k[0] == "real" for (k, _, _), w in zip(cand, inside) if w), "real,",
      sum(k[0] == "syn" for (k, _, _), w in zip(cand, inside) if w), "synthetic")
z = lambda a: (a - a.mean()) / a.std()
# Images outside the window are ranked below every image inside it, so they are used only where
# a device has fewer than three of its own inside.
score = z(iou) + z(dsc) - 100.0 * (~inside)
ranked = sorted(zip(score, cand), key=lambda t: -t[0])
PER_DEV, NROW = 3, 3
real = [c for d in DEVICES for c in [c for s, c in ranked if c[0][0] == "real" and c[0][2] == d][:PER_DEV]]
syn = [c for s, c in ranked if c[0][0] == "syn"][:NROW * NCOL]
# PICKS=<a medsam_figure_picks.json> draws exactly those images instead of re-selecting, which is
# how an earlier version of the figure is reproduced on a new probe run (the image order is the
# same in every build of the arrays, so an index means the same radiograph).
PICKS = os.environ.get("PICKS", "")
if PICKS:
    d0 = json.load(open(PICKS))
    real = [((("real"), r["idx"], r["source"]), 0.0, 0.0) for r in d0["real"]]
    syn = [((("syn"), r["idx"], "PanoDiff-SR"), 0.0, 0.0) for r in d0["syn"]]
    print("picks taken from", PICKS)
assert len(real) == NROW * NCOL and len(syn) == NROW * NCOL, (len(real), len(syn))
json.dump({"criterion": f"reused from {PICKS}" if PICKS else "per-region protocol, rank by z(mean predicted IoU) + z(mean Dice to U-Net "
                        "reference) among complete dentitions (reference tooth region spans >=55% of the width, "
                        "covers the span without gaps in both arches, >=3 tooth regions, area within the 30th-95th "
                        "percentile of the real set; no letterboxed image); real: best three per device; "
                        "synthetic: top twelve",
           "real": [{"idx": c[0][1], "source": c[0][2], "pred_iou": c[1], "dice_ref": c[2]} for c in real],
           "syn": [{"idx": c[0][1], "pred_iou": c[1], "dice_ref": c[2]} for c in syn]},
          open(f"{POUT}/medsam_figure_picks.json", "w"), indent=1)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = load_medsam(dev)
imgs = {"real": np.load(f"{PDATA}/real_img.npy", mmap_mode="r"), "syn": np.load(f"{PDATA}/syn_img.npy", mmap_mode="r")}
refs = {"real": np.load(f"{PREF}/ref_real.npy", mmap_mode="r"),
        "syn": np.load(f"{PREF}/ref_syn.npy", mmap_mode="r")}

# Six rows: three of real, three of synthetic; a label strip only above the first row of each set.
ROWS = [("Real radiographs (three per device)", real[:NCOL])] + [(None, real[r * NCOL:(r + 1) * NCOL]) for r in range(1, NROW)] \
     + [("PanoDiff-SR", syn[:NCOL])] + [(None, syn[r * NCOL:(r + 1) * NCOL]) for r in range(1, NROW)]
strip = [HEADER_PT if lab else 2.0 for lab, _ in ROWS]
h_pt = sum(strip) + len(ROWS) * PANEL_H
fig = plt.figure(figsize=(W_PT * PT, h_pt * PT))
plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
for r, (label, picks) in enumerate(ROWS):
    top = h_pt - sum(strip[:r + 1]) - r * PANEL_H          # top edge of this row's panels
    if label:
        fig.text(0.002, (top + HEADER_PT / 2) / h_pt, label, ha="left", va="center", fontsize=7.2)
    for c, (key, mi, md) in enumerate(picks):
        which, i, src = key
        img = np.asarray(imgs[which][i]); ref = np.asarray(refs[which][i]) > 0
        comps = boxes_from_components(ref)
        masks, ious = prompt(sam, embed(sam, img, dev), [b for b, _ in comps], dev)
        ax = fig.add_axes([c / NCOL, (top - PANEL_H) / h_pt, 1 / NCOL, PANEL_H / h_pt])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bilinear", aspect="auto")
        for (x0, y0, x1, y1), _ in comps:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ffd21f", lw=0.45))
        union = np.zeros((H, W), bool)
        for m in masks:
            union |= m
        ax.contour(union.astype(float), levels=[0.5], colors=["#19d3ff"], linewidths=0.5)
        ax.text(0.985, 0.05, f"{np.mean(ious):.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        if which == "real":
            ax.text(0.015, 0.05, src, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.6")
os.makedirs(FIGS, exist_ok=True)
fig.savefig(f"{FIGS}/medsam_probe.pdf", dpi=600)
fig.savefig(f"{FIGS}/medsam_probe.png", dpi=170)
print("wrote", f"{FIGS}/medsam_probe.pdf", "real:", [(c[0][2], c[0][1]) for c in real], "syn:", [c[0][1] for c in syn])
