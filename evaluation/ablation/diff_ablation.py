"""Diffusion-stage ablations for reviewer R2.4: EMA and sampler budget.

Both are exact, no retraining: every diffusion checkpoint stores 'model_state'
and 'ema_model_state', and the number of DDIM steps is a sampling-time choice.
The sampling call matches generate.py exactly (eta=1.0, cosine schedule, the
project's own DDIMScheduler), so the reference variant reproduces the sampler
used for the paper's synthetic set.

Each variant writes its samples to a folder; fid_is.py then scores every folder
against the same real reference set with the same metric implementation the
paper used (torchmetrics FID, normalize=True).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

P = WORK + "/project-files/PanoDiff/Project"
sys.path.insert(0, P + "/sd")

from simple_diffusion.scheduler import DDIMScheduler        # noqa: E402
from simple_diffusion.model import UNet                     # noqa: E402

CKPT = P + "/sd/old_results/old_trained_models/ddpm-model-ep110.pth"

# name -> (weight key, beta schedule, DDIM steps)
VARIANTS = {
    "ema-250":   ("ema_model_state", "cosine", 250),   # the paper's configuration
    "noema-250": ("model_state",     "cosine", 250),
    "ema-100":   ("ema_model_state", "cosine", 100),
    "ema-50":    ("ema_model_state", "cosine", 50),
    "ema-25":    ("ema_model_state", "cosine", 25),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    key, schedule, steps = VARIANTS[args.variant]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(3, image_size=(args.resolution, 2 * args.resolution),
                 hidden_dims=[64, 128, 256, 512], use_flash_attn=False)
    sd = torch.load(args.ckpt, map_location="cpu")[key]
    model.load_state_dict(sd, strict=False)
    model.eval().to(dev)

    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule=schedule)

    out = os.path.join(args.outdir, args.variant)
    os.makedirs(out, exist_ok=True)

    made, t0 = 0, time.time()
    seed = 0
    while made < args.n:
        bs = min(args.batch, args.n - made)
        gen = torch.manual_seed(seed)
        seed += 1
        with torch.no_grad():
            res = sched.generate(model, batch_size=bs, generator=gen, eta=1.0,
                                 num_inference_steps=steps, device=dev)
        imgs = (res["sample"] * 255).round().astype("uint8")
        for im in imgs:
            Image.fromarray(im).save(os.path.join(out, "%05d.png" % made))
            made += 1
        print("  %s %d/%d  %.1f s/img" % (args.variant, made, args.n,
                                          (time.time() - t0) / made), flush=True)

    dt = time.time() - t0
    with open(os.path.join(args.outdir, args.variant + ".time"), "w") as f:
        f.write("%s %d %d %.4f\n" % (args.variant, steps, args.n, dt / args.n))
    print("DONE %s steps=%d n=%d %.4f s/img" % (args.variant, steps, args.n, dt / args.n))


if __name__ == "__main__":
    main()
