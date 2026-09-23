"""t-SNE of every generative arm in the paper, in the feature space FID uses.

Why this exists: the paper's two older t-SNE figures cannot be regenerated -- the
script that computed them is gone and their coordinates were scraped back out of
the PDFs. This recomputes an embedding from the images themselves, for all arms.

Stages (run all by default; each caches its output so re-plotting needs no GPU):
  embed  -- N images per arm, deterministic sample; InceptionV3 pool3 features via
            torchmetrics' FrechetInceptionDistance(normalize=True).inception, i.e.
            exactly the network and preprocessing behind every FID in the paper.
            ResNet-50 avgpool features are extracted in the same pass (Figure 8's
            extractor) so either space can be plotted later without a GPU.
  tsne   -- one embedding per panel: PCA to 50 dims, then t-SNE.
  plot   -- panels in the visual style of figure-sources/make_tsne_gt_pd.py.

Usage: python3 make_tsne_arms.py [--stages embed,tsne,plot] [--feature inception|resnet50]
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, json, os
import numpy as np

LEGACY = WORK + "/project-files/PanoDiff/Syn-Calc-FID&IS"
DIRECT = WORK + "/new-baselines-generation/samples"
OUT = WORK + "/new-files/analysis/tsne_arms"
FIG = WORK + "/new-files/overleaf/figures/fig19-tsne-arms"
SWFT = WORK + "/swinir-ft-eval/out"   # PR-fine-tuned SwinIR
RESNET = _os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth")

N_PER_ARM = 1500
SEED = 0
# The FastGAN+HAT-SR folder holds 100 extra samples (seeds 7244-7343); keep the 7243 seeds
# that the other FastGAN arms and Table 9 use.
CAP = {"gan_hat": 7243}

# key -> (display label, folder). Legacy folder numbers are Table 7's Index column.
ARMS = {
    "lrgt":       ("Real LR (1)",              f"{LEGACY}/1-TrainingDataInLR"),
    "lrpd":       ("PanoDiff LR (2)",          f"{LEGACY}/2-DiffLR"),
    "lrgan":      ("FastGAN LR (3)",           f"{LEGACY}/3-GANsLR"),
    "hrpd":       ("PanoDiff + HAT-SR (4)",    f"{LEGACY}/4-DiffHATSR"),
    "gan_hat":    ("FastGAN + HAT-SR (5)",     f"{LEGACY}/5-GANsHATSR"),
    "pd_swin":    ("PanoDiff + SwinIR (6)",    f"{SWFT}/6ft-DiffSwinIRft"),
    "gan_swin":   ("FastGAN + SwinIR (7)",     f"{SWFT}/7ft-GANsSwinIRft"),
    "gt_hat":     ("Real LR + HAT-SR (8)",     f"{LEGACY}/8-TrainHATSR"),
    "gt_swin":    ("Real LR + SwinIR (9)",     f"{SWFT}/9ft-TrainSwinIRft"),
    "hrgt":       ("Real HR (10)",             f"{LEGACY}/10-TrainDataInHR"),
    "sg2ada":     ("StyleGAN2-ADA",            f"{DIRECT}/sg2ada"),
    "sg3":        ("StyleGAN3",                f"{DIRECT}/sg3"),
    "adm":        ("ADM",                      f"{DIRECT}/adm"),
    "ldm":        ("LDM",                      f"{DIRECT}/ldm"),
    "panodiff_hr":("PanoDiff-HR",              f"{DIRECT}/panodiff_hr"),
    "fastgan_hr": ("FastGAN-HR",               f"{DIRECT}/fastgan_lite64"),
}

# Each panel is embedded separately, as Figure 8's two panels were. The reference
# (real) arm is listed first; PanoDiff-SR second so it is drawn on top.
PANELS = [
    ("lr",  "Low-resolution generators ($256\\times128$)", ["lrgt", "lrpd", "lrgan"]),
    ("sr",  "Super-resolution variants ($1024\\times512$)",
            ["hrgt", "hrpd", "pd_swin", "gan_hat", "gan_swin", "gt_hat", "gt_swin"]),
    ("gen", "Generators at full resolution ($1024\\times512$)",
            ["hrgt", "hrpd", "sg2ada", "sg3", "adm", "ldm", "panodiff_hr", "fastgan_hr"]),
]


def list_images(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))


def sample(folder, key):
    files = list_images(folder)[:CAP.get(key)]
    # Seed derived from the arm key, so adding an arm never reshuffles another.
    rng = np.random.default_rng(SEED + sum(map(ord, key)))
    idx = np.sort(rng.choice(len(files), size=min(N_PER_ARM, len(files)), replace=False))
    return [files[i] for i in idx]


# ------------------------------------------------------------------ embed ---
def embed():
    import torch, torchvision
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchmetrics.image.fid import FrechetInceptionDistance
    from PIL import Image
    import torchvision.transforms as T

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, flush=True)
    inception = FrechetInceptionDistance(normalize=True).inception.to(dev).eval()
    resnet = torchvision.models.resnet50()
    # torchvision's own release file, legacy .tar format -> weights_only must be off.
    resnet.load_state_dict(torch.load(RESNET, map_location="cpu", weights_only=False))
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    class Ds(Dataset):
        def __init__(self, folder, names):
            self.folder, self.names, self.t = folder, names, T.ToTensor()
        def __len__(self):
            return len(self.names)
        def __getitem__(self, i):
            return self.t(Image.open(os.path.join(self.folder, self.names[i])).convert("RGB"))

    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for key, (label, folder) in ARMS.items():
        dst = os.path.join(OUT, f"feat_{key}.npz")
        names = sample(folder, key)
        manifest[key] = {"label": label, "folder": folder, "n": len(names)}
        if os.path.exists(dst):
            print(f"[{key}] cached", flush=True)
            continue
        fi, fr = [], []
        with torch.no_grad():
            for b in DataLoader(Ds(folder, names), batch_size=64, num_workers=6, pin_memory=True):
                b = b.to(dev)
                # Same conversion torchmetrics applies with normalize=True.
                fi.append(inception((b * 255).byte()).float().cpu())
                # ResNet-50: aspect-preserving resize to 224x448, ImageNet norm.
                r = F.interpolate(b, size=(224, 448), mode="bilinear", align_corners=False)
                fr.append(resnet((r - mean) / std).float().cpu())
        np.savez_compressed(dst, inception=torch.cat(fi).numpy(),
                            resnet50=torch.cat(fr).numpy(), names=np.array(names))
        print(f"[{key}] {len(names)} images -> {dst}", flush=True)
    json.dump({"n_per_arm": N_PER_ARM, "seed": SEED, "arms": manifest},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)


# ------------------------------------------------------------------- tsne ---
def tsne(feature):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    params = {"pca_dims": 50, "perplexity": 30, "init": "pca",
              "learning_rate": "auto", "max_iter": 1500, "random_state": SEED}
    rp = os.path.join(OUT, f"tsne_{feature}_report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {"feature": feature, "params": params, "panels": {}}
    only = os.environ.get("TSNE_PANELS")
    for tag, _, keys in PANELS:
        if only and tag not in only.split(","):
            continue
        X, lab = [], []
        for k in keys:
            f = np.load(os.path.join(OUT, f"feat_{k}.npz"))[feature]
            X.append(f); lab += [k] * len(f)
        X = np.vstack(X).astype(np.float64)
        lab = np.array(lab)
        Xp = PCA(n_components=params["pca_dims"], random_state=SEED).fit_transform(X)
        Y = TSNE(n_components=2, perplexity=params["perplexity"], init=params["init"],
                 learning_rate=params["learning_rate"], max_iter=params["max_iter"],
                 random_state=SEED).fit_transform(Xp)
        np.savez_compressed(os.path.join(OUT, f"tsne_{feature}_{tag}.npz"), Y=Y, labels=lab)
        # Distances measured in the ORIGINAL feature space, where they mean something;
        # t-SNE distances between clusters are not interpretable.
        ref = keys[0]
        mu_ref = X[lab == ref].mean(0)
        dists = {k: float(np.linalg.norm(X[lab == k].mean(0) - mu_ref)) for k in keys[1:]}
        report["panels"][tag] = {"arms": keys, "reference": ref,
                                 "feature_space_centroid_distance": dists}
        print(f"[tsne:{tag}] {len(Y)} points; distances to {ref}: "
              + ", ".join(f"{k}={v:.2f}" for k, v in dists.items()), flush=True)
    json.dump(report, open(os.path.join(OUT, f"tsne_{feature}_report.json"), "w"), indent=2)


# ------------------------------------------------------------------- plot ---
def plot(feature):
    """Upright full-page layout: one row per panel, scatter on the left and its legend
    on the right, sized to the paper's 390 pt text width so nothing is shrunk later."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter

    PT = 1 / 72.27
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 7.2, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })
    # Real keeps Figure 8's blue and PanoDiff its red; the rest are Okabe-Ito.
    COL = {"hrgt": "#1b6ca8", "lrgt": "#1b6ca8", "hrpd": "#d1495b", "lrpd": "#d1495b",
           "lrgan": "#E69F00", "gan_hat": "#E69F00", "sg2ada": "#009E73",
           "pd_swin": "#CC79A7", "adm": "#CC79A7", "gan_swin": "#56B4E9", "ldm": "#56B4E9",
           "gt_hat": "#7a7a7a", "panodiff_hr": "#7a7a7a", "gt_swin": "#8c564b",
           "fastgan_hr": "#8c564b", "sg3": "#6a3d9a"}
    fig = plt.figure(figsize=(390 * PT, 470 * PT))
    gs = fig.add_gridspec(len(PANELS), 2, width_ratios=[1.75, 1], hspace=0.42, wspace=0.04,
                          left=0.07, right=0.995, top=0.965, bottom=0.05)
    rng = np.random.default_rng(SEED)
    for row, (tag, title, keys) in enumerate(PANELS):
        ax = fig.add_subplot(gs[row, 0])
        d = np.load(os.path.join(OUT, f"tsne_{feature}_{tag}.npz"))
        Y, lab = d["Y"], d["labels"]
        lo, hi = Y.min(0) - 4, Y.max(0) + 4
        gx, gy = np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200)
        order = [keys[0]] + keys[2:] + [keys[1]]      # real first, PanoDiff on top
        for k in order:
            P_ = Y[lab == k]
            idx = rng.choice(len(P_), size=min(600, len(P_)), replace=False)
            ax.scatter(P_[idx, 0], P_[idx, 1], s=1.6, c=COL[k], alpha=0.35, linewidths=0,
                       rasterized=True)
        for k in order:
            P_ = Y[lab == k]
            Hh, _, _ = np.histogram2d(P_[:, 0], P_[:, 1], bins=[gx, gy])
            Hh = gaussian_filter(Hh.T, 4.0)
            if Hh.max() > 0:
                ax.contour(0.5 * (gx[1:] + gx[:-1]), 0.5 * (gy[1:] + gy[:-1]), Hh,
                           levels=[0.5 * Hh.max()], colors=COL[k],
                           linewidths=1.3 if k in keys[:2] else 0.8)
            ax.plot(*P_.mean(0), marker="X", ms=5.5, mfc=COL[k], mec="k", mew=0.6, zorder=6)
        ax.set_title(f"({chr(97 + row)}) {title}", loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t-SNE 1", labelpad=1); ax.set_ylabel("t-SNE 2", labelpad=1)
        lg = fig.add_subplot(gs[row, 1]); lg.axis("off")
        h = [Line2D([], [], marker="o", ls="", color=COL[k], ms=5, label=ARMS[k][0]) for k in keys]
        h.append(Line2D([], [], marker="X", ls="", mfc="0.6", mec="k", mew=0.6, ms=5.5,
                        label="Centroid"))
        h.append(Line2D([], [], ls="-", color="0.4", lw=1.0, label="50% density contour"))
        lg.legend(handles=h, loc="center left", frameon=False, handletextpad=0.3,
                  labelspacing=0.45, borderaxespad=0)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"tsne_arms_{feature}.pdf")
    fig.savefig(out, dpi=400)
    fig.savefig(out.replace(".pdf", ".png"), dpi=170)
    plt.close(fig)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="embed,tsne,plot")
    ap.add_argument("--feature", default="inception", choices=["inception", "resnet50"])
    a = ap.parse_args()
    st = a.stages.split(",")
    if "embed" in st: embed()
    if "tsne" in st: tsne(a.feature)
    if "plot" in st: plot(a.feature)
