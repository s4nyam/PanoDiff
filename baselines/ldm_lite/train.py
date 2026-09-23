"""Latent-diffusion baseline: stage 1 trains the VAE, stage 2 the latent DDPM.

This is the third diffusion baseline for R2.5 and the one that matters most
scientifically: latent compression is the *other* way to make high-resolution
diffusion affordable, so it competes directly with the two-stage design rather
than merely being another pixel-space model.

  --stage vae : KL autoencoder, 512x1024 -> 4x64x128
  --stage ldm : the PanoDiff UNet (proven here) run on those latents, using the
                same cosine schedule, L1 objective and DDIM sampler as PanoDiff,
                so the only difference from the PanoDiff-HR arm is *where* the
                diffusion happens.

Everything runs in bf16 with gradient clipping: fp16 silently destabilised both
Medfusion stages (its VAE latents reached +/-80,000, its LDM went NaN).
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

B = WORK + "/new-baselines-generation"
sys.path.insert(0, B)
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
sys.path.insert(0, WORK + "/project-files/PanoDiff/Project/sd")
from ldm_lite.vae import VAE                                   # noqa: E402
from simple_diffusion.model.unet import UNet                   # noqa: E402
from simple_diffusion.scheduler import DDIMScheduler           # noqa: E402

SEED, H, W = 42, 512, 1024
LH, LW, ZCH = H // 8, W // 8, 4


def setup():
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        r, ws, lr = (int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]),
                     int(os.environ.get("SLURM_LOCALID", 0)))
    else:
        r, ws, lr = 0, 1, 0
    torch.cuda.set_device(lr)
    if ws > 1:
        dist.init_process_group("nccl", rank=r, world_size=ws)
    return r, ws, lr


class PRData(Dataset):
    def __init__(self, listfile):
        self.files = [l.strip() for l in open(listfile) if l.strip()]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            im = im.convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), Image.LANCZOS)
            a = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
        return a.permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vae", "ldm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vae-ckpt", default=f"{B}/runs/ldm_vae/latest.pth")
    args = ap.parse_args()

    rank, world, local = setup()
    main_p = rank == 0
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("cuda"); os.makedirs(args.out, exist_ok=True)

    ds = PRData(f"{B}/data/train_hr.txt")
    samp = DistributedSampler(ds, world, rank, shuffle=True, seed=SEED) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch, sampler=samp, shuffle=(samp is None),
                    num_workers=5, pin_memory=True, drop_last=True, persistent_workers=True)

    vae = VAE(z_ch=ZCH).to(dev)
    if args.stage == "vae":
        model, params = vae, vae.parameters()
    else:
        c = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
        vae.load_state_dict(c["model"]); vae.eval().requires_grad_(False)
        # the proven PanoDiff denoiser, operating on the 4x64x128 latent grid
        model = UNet(ZCH, image_size=(LH, LW), hidden_dims=[128, 256, 384, 512],
                     use_flash_attn=False).to(dev)
        model.conv_out = torch.nn.Conv2d(128, ZCH, 1).to(dev)
        params = model.parameters()
        sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
        zs = c.get("latent_scale", 1.0)
        if main_p: print(f"latent scale {zs:.4f}", flush=True)

    if main_p:
        print(f"stage={args.stage} params {sum(p.numel() for p in model.parameters())/1e6:.1f} M "
              f"| global batch {world*args.batch}", flush=True)
    if world > 1:
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t.data, src=0)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    def allreduce(m):
        if world == 1: return
        for p in m.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(world)

    ck = os.path.join(args.out, "latest.pth"); start = 0
    if os.path.exists(ck):
        c2 = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c2["model"]); opt.load_state_dict(c2["opt"]); start = c2["epoch"] + 1
        if main_p: print(f"resumed ep {start}", flush=True)

    t0 = time.time(); gstep = 0
    for ep in range(start, args.epochs):
        if samp is not None: samp.set_epoch(ep)
        model.train(); run = 0.0; n = 0
        for x in dl:
            x = x.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if args.stage == "vae":
                    _, loss, rec, kl = model(x)
                else:
                    with torch.no_grad():
                        z = vae.encode(x) * zs
                    noise = torch.randn_like(z)
                    t = torch.randint(0, 1000, (z.shape[0],), device=dev).long()
                    zt = sched.add_noise(z, noise, t)
                    loss = F.l1_loss(model(zt, t)["sample"], noise)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            allreduce(model); opt.step()
            run += loss.item(); n += 1; gstep += 1
            if main_p and gstep % 200 == 0:
                el = time.time() - t0
                extra = f" rec {rec:.4f} kl {kl:.3f}" if args.stage == "vae" else ""
                print(f"ep {ep} step {gstep} loss {run/n:.4f}{extra} {el/3600:.2f}h", flush=True)
        if main_p:
            blob = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}
            if args.stage == "vae":
                # latent scale so the diffusion sees roughly unit variance
                model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    zz = torch.cat([model.encode(next(iter(dl)).to(dev)) for _ in range(2)])
                blob["latent_scale"] = float(1.0 / (zz.float().std() + 1e-8))
                model.train()
            torch.save(blob, ck)
            json.dump({"epoch": ep, "loss": run/max(n,1), "hours": (time.time()-t0)/3600},
                      open(os.path.join(args.out, "progress.json"), "w"), indent=1)
    if main_p: print(f"DONE {args.stage} in {(time.time()-t0)/3600:.2f} h", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
