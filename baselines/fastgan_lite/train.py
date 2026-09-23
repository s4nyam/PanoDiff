"""Train the reimplemented FastGAN directly at 1024x512.

This is the matched control for the paper's existing FastGAN+HAT-SR arm: same
architecture family, but generating the full-resolution radiograph in one shot
instead of a 256x128 seed that is then super-resolved. Together with PanoDiff-HR
it gives a direct-vs-modular test on both the GAN and the diffusion side.

Recipe from the paper: hinge adversarial loss, Adam(0.5, 0.999) at 2e-4, and the
self-supervised reconstruction loss on the discriminator's decoders (applied to
REAL images only, as in the paper -- it regularises D, it is not a generator
objective). Budget is the shared 110 epochs.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

B = WORK + "/new-baselines-generation"
sys.path.insert(0, B)
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
from fastgan_lite.networks import Generator, Discriminator    # noqa: E402
from ldm_lite.train import PRData                             # noqa: E402

SEED, H, W, NZ = 42, 512, 1024, 256


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--recon-w", type=float, default=1.0)
    args = ap.parse_args()

    rank, world, local = setup()
    mp = rank == 0
    torch.manual_seed(SEED + rank); np.random.seed(SEED + rank)
    dev = torch.device("cuda"); os.makedirs(args.out, exist_ok=True)

    ds = PRData(f"{B}/data/train_hr.txt")
    samp = DistributedSampler(ds, world, rank, shuffle=True, seed=SEED) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch, sampler=samp, shuffle=(samp is None),
                    num_workers=5, pin_memory=True, drop_last=True, persistent_workers=True)

    G = Generator(nz=NZ, out_hw=(H, W)).to(dev)
    D = Discriminator(in_hw=(H, W)).to(dev)
    G_ema = Generator(nz=NZ, out_hw=(H, W)).to(dev).eval().requires_grad_(False)
    G_ema.load_state_dict(G.state_dict())
    if mp:
        print(f"G {sum(p.numel() for p in G.parameters())/1e6:.1f} M | "
              f"D {sum(p.numel() for p in D.parameters())/1e6:.1f} M | "
              f"native {H}x{W} | global batch {world*args.batch}", flush=True)
    if world > 1:
        for m in (G, D, G_ema):
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t.data, src=0)

    def allreduce(m):
        if world == 1: return
        for p in m.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(world)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    ck = os.path.join(args.out, "latest.pth"); start = 0; gstep = 0
    if os.path.exists(ck):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        G.load_state_dict(c["G"]); D.load_state_dict(c["D"]); G_ema.load_state_dict(c["G_ema"])
        optG.load_state_dict(c["optG"]); optD.load_state_dict(c["optD"])
        start, gstep = c["epoch"] + 1, c["gstep"]
        if mp: print(f"resumed ep {start}", flush=True)

    total = args.epochs * len(dl); t0 = time.time()
    for ep in range(start, args.epochs):
        if samp is not None: samp.set_epoch(ep)
        for real in dl:
            real = real.to(dev, non_blocking=True); n = real.shape[0]

            # ---- D: hinge loss + self-supervised reconstruction on REALS ----
            with torch.no_grad():
                fake = G(torch.randn(n, NZ, device=dev))
            d_real, r16, r8 = D(real, recon=True)
            d_fake = D(fake)
            tgt16 = F.interpolate(real, size=r16.shape[-2:], mode="bilinear", align_corners=False)
            tgt8  = F.interpolate(real, size=r8.shape[-2:],  mode="bilinear", align_corners=False)
            loss_rec = F.l1_loss(r16, tgt16) + F.l1_loss(r8, tgt8)
            loss_d = (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                      + args.recon_w * loss_rec)
            optD.zero_grad(set_to_none=True); loss_d.backward(); allreduce(D); optD.step()

            # ---- G: hinge ----
            fake = G(torch.randn(n, NZ, device=dev))
            loss_g = -D(fake).mean()
            optG.zero_grad(set_to_none=True); loss_g.backward(); allreduce(G); optG.step()

            with torch.no_grad():
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.mul_(0.999).add_(pg, alpha=0.001)
                for be, bg in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(bg)
            gstep += 1
            if mp and gstep % 200 == 0:
                el = time.time() - t0
                print(f"ep {ep} step {gstep}/{total} d {loss_d.item():.3f} g {loss_g.item():.3f} "
                      f"rec {loss_rec.item():.4f} {el/3600:.2f}h "
                      f"eta {el/max(gstep,1)*(total-gstep)/3600:.1f}h", flush=True)
        if mp:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "epoch": ep, "gstep": gstep}, ck)
            json.dump({"epoch": ep, "gstep": gstep, "hours": (time.time()-t0)/3600},
                      open(os.path.join(args.out, "progress.json"), "w"), indent=1)
    if mp: print(f"DONE {args.epochs} epochs in {(time.time()-t0)/3600:.2f} h", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
