"""MedSAM probing of real vs synthetic panoramic radiographs.

MedSAM (Ma et al., Nat. Commun. 2024) is a frozen, promptable segmenter trained on ~1.5M
medical image-mask pairs, none of them ours. The idea: give it the SAME prompts on real and on
PanoDiff-SR radiographs and compare what it does. If the synthetic images carry realistic
dental anatomy, a foundation model that has never seen either set should segment teeth in them
the way it segments teeth in real ones -- same confidence, same mask shapes, same agreement with
an independent tooth model. Three prompt protocols, from model-free to anatomy-driven:

  fixed     one box over the dental region at the same fixed image coordinates for every image
            (no model of ours involved at all)
  arch      one box around the tooth region predicted by the binary U-Net of the downstream
            study (trained on all 1888 real labels); identical procedure on both sets
  teeth     one box per connected tooth component of that same U-Net prediction, so MedSAM
            segments each tooth; a per-tooth score, plus how many teeth it was asked for

Per prompt we record MedSAM's own predicted IoU (its confidence head), the mask area, the
number of connected components, the solidity of the mask, and the Dice between MedSAM's mask
and the U-Net reference (for real images also against the manual ground truth, which is the
sanity check that MedSAM works on panoramics at all). Written as one CSV row per prompt.

Weights and preprocessing: the authors' release (medsam_vit_b_official.pth) loaded through the
authors' segment_anything definition; squash-resize to 1024x1024 and per-image min-max scaling,
exactly as MedSAM_Inference.py does. Boxes are given in that 1024x1024 space. fp32 throughout.

Sharded: --shard k --nshards n processes every n-th image, so 8 GCDs cover both sets in one job.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

E = WORK
D = f"{E}/downstream-seg/data"
CKPT = WORK + "/weights/medsam_vit_b_official.pth"
H, W, S = 512, 1024, 1024
FIXED_BOX = (0.15, 0.30, 0.85, 0.80)        # x0, y0, x1, y1 as image fractions: the dental region
MIN_TOOTH_PX = 400                         # components smaller than this are not teeth at 1024x512


def load_medsam(dev):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=None)
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    msg = sam.load_state_dict(st.get("model", st), strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, msg
    return sam.to(dev).eval()


@torch.no_grad()
def embed(sam, img_u8, dev):
    x = torch.from_numpy(img_u8).float().to(dev)[None, None].expand(1, 3, H, W)
    x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
    x = (x - x.min()) / (x.max() - x.min()).clamp(min=1e-8)
    return sam.image_encoder(x)


@torch.no_grad()
def prompt(sam, emb, boxes_xyxy, dev):
    """boxes in image pixels -> (masks [n,H,W] bool, predicted IoU [n])."""
    b = torch.tensor(boxes_xyxy, dtype=torch.float32, device=dev)
    b[:, [0, 2]] *= S / W; b[:, [1, 3]] *= S / H
    sparse, dense = sam.prompt_encoder(points=None, boxes=b, masks=None)
    low, iou = sam.mask_decoder(image_embeddings=emb,          # [1,...]: the decoder repeats it per box
                                image_pe=sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                multimask_output=False)
    lg = F.interpolate(low, size=(S, S), mode="bilinear", align_corners=False)
    lg = F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    return (lg > 0).cpu().numpy(), iou[:, 0].cpu().numpy()


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def shape_stats(m):
    lab, n = ndimage.label(m)
    area = int(m.sum())
    if area == 0:
        return dict(area=0, n_comp=0, solidity=0.0)
    ys, xs = np.nonzero(m)
    hull = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return dict(area=area, n_comp=int(n), solidity=float(area / hull))


def boxes_from_components(ref):
    lab, n = ndimage.label(ref)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < MIN_TOOTH_PX:
            continue
        y0, x0 = sl[0].start, sl[1].start
        out.append(([x0, y0, sl[1].stop, sl[0].stop], (sl, comp)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=f"{E}/medsam-probe/out")
    ap.add_argument("--data", default=D, help="directory with real_img/real_msk/syn_img.npy and meta.json")
    ap.add_argument("--ref", default=f"{E}/medsam-probe/data", help="directory with ref_real.npy and ref_syn.npy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = load_medsam(dev)

    meta = json.load(open(f"{a.data}/meta.json"))
    real = np.load(f"{a.data}/real_img.npy", mmap_mode="r"); gt = np.load(f"{a.data}/real_msk.npy", mmap_mode="r")
    syn = np.load(f"{a.data}/syn_img.npy", mmap_mode="r")
    # Reference tooth masks from the downstream U-Net (all labels): real images -> its prediction
    # them and on the synthetic set, both dumped by one run (train.py --dump-pred).
    ref_real = np.load(f"{a.ref}/ref_real.npy", mmap_mode="r")
    ref_syn = np.load(f"{a.ref}/ref_syn.npy", mmap_mode="r")

    items = [("real", i, meta[i]["dataset"]) for i in range(len(real))] + \
            [("syn", i, "PanoDiff-SR") for i in range(len(syn))]
    items = items[a.shard::a.nshards]
    if a.limit:
        items = items[:a.limit]

    fx = [int(FIXED_BOX[0] * W), int(FIXED_BOX[1] * H), int(FIXED_BOX[2] * W), int(FIXED_BOX[3] * H)]
    path = f"{a.out}/probe_shard{a.shard:02d}.csv"
    fh = open(path, "w", newline=""); w = csv.writer(fh)
    w.writerow(["set", "idx", "source", "protocol", "k", "pred_iou", "area", "n_comp", "solidity",
                "dice_ref", "dice_gt", "ref_area"])
    t0 = time.time()
    for n, (which, i, src) in enumerate(items):
        img = np.asarray(real[i] if which == "real" else syn[i])
        ref = np.asarray(ref_real[i] if which == "real" else ref_syn[i]) > 0
        g = np.asarray(gt[i]) > 0 if which == "real" else None
        emb = embed(sam, img, dev)

        # fixed box
        m, iou = prompt(sam, emb, [fx], dev)
        st = shape_stats(m[0])
        w.writerow([which, i, src, "fixed", 0, float(iou[0]), st["area"], st["n_comp"], st["solidity"],
                    dice(m[0], ref), dice(m[0], g) if g is not None else "", int(ref.sum())])
        # arch box around the reference tooth region
        if ref.any():
            ys, xs = np.nonzero(ref)
            m, iou = prompt(sam, emb, [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dev)
            st = shape_stats(m[0])
            w.writerow([which, i, src, "arch", 0, float(iou[0]), st["area"], st["n_comp"], st["solidity"],
                        dice(m[0], ref), dice(m[0], g) if g is not None else "", int(ref.sum())])
        # one box per tooth component
        comps = boxes_from_components(ref)
        if comps:
            m, iou = prompt(sam, emb, [c[0] for c in comps], dev)
            for k, (mk, ik, (box, (sl, comp))) in enumerate(zip(m, iou, comps)):
                full = np.zeros_like(ref); full[sl] = comp
                st = shape_stats(mk)
                w.writerow([which, i, src, "teeth", k, float(ik), st["area"], st["n_comp"], st["solidity"],
                            dice(mk, full), dice(mk, g & full) if g is not None else "", int(full.sum())])
        if (n + 1) % 50 == 0:
            fh.flush(); print(f"shard {a.shard}: {n + 1}/{len(items)} {(time.time() - t0) / (n + 1):.2f}s/img", flush=True)
    fh.close()
    print("wrote", path, len(items), "images", flush=True)


if __name__ == "__main__":
    main()
