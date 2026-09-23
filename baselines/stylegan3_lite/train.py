"""Train StyleGAN3-T on the panoramic radiographs, native 512x1024.

Recipe = NVIDIA's `train.py --cfg=stylegan3-t` (repos/stylegan3/train.py):
  * G: StyleGAN3-T (2-layer mapping, alias-free synthesis), lr 0.0025, Adam(0, 0.99),
    magnitude EMA beta 0.5 ** (batch / 20k); no path-length reg, no style mixing
  * D: StyleGAN2 residual discriminator, lr 0.002, lazy R1 every 16 steps
  * R1 gamma 8.2 = NVIDIA's value for StyleGAN3-T at 512 (AFHQv2) and the value used
    for the StyleGAN2-ADA arm; this network is a 512 network on a 2:1 canvas
  * ADA, target 0.6; G EMA with ema_kimg = batch * 10 / 32 and ramp-up 0.05
  * global batch 32 (32 ranks x 1)
The training loop is stylegan2_lite/train.py's, so the two StyleGAN arms share data
loading, ADA, R1 and gradient averaging. Budget: 110 epochs x 7243 images.
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
from stylegan3_lite.networks import Generator                     # noqa: E402
from stylegan2_lite.networks import Discriminator                 # noqa: E402
from stylegan2_lite.augment import AugmentPipe, ADAController     # noqa: E402
from stylegan2_lite.train import PRData, setup_dist, H, W         # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--batch", type=int, default=1, help="per rank")
    ap.add_argument("--glr", type=float, default=2.5e-3)
    ap.add_argument("--dlr", type=float, default=2.0e-3)
    ap.add_argument("--r1-gamma", type=float, default=8.2)
    ap.add_argument("--d-reg-every", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    args = ap.parse_args()

    rank, world, local = setup_dist()
    main_proc = rank == 0
    torch.manual_seed(SEED + rank); np.random.seed(SEED + rank)
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    ds = PRData(f"{B}/data/train_hr.txt")
    samp = DistributedSampler(ds, world, rank, shuffle=True, seed=SEED) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch, sampler=samp, shuffle=(samp is None),
                    num_workers=5, pin_memory=True, drop_last=True, persistent_workers=True)

    global_batch = world * args.batch
    mag_beta = 0.5 ** (global_batch / 20e3)
    torch.manual_seed(SEED)                              # identical init on every rank
    G = Generator(img_resolution=(H, W), magnitude_ema_beta=mag_beta).to(dev)
    D = Discriminator(img_resolution=(H, W)).to(dev)
    G_ema = Generator(img_resolution=(H, W), magnitude_ema_beta=mag_beta).to(dev).eval().requires_grad_(False)
    G_ema.load_state_dict(G.state_dict())
    torch.manual_seed(SEED + rank)
    aug = AugmentPipe().to(dev)
    ada = ADAController(aug, target=0.6, interval=4, batch=global_batch)

    if main_proc:
        print(f"StyleGAN3-T | G {sum(p.numel() for p in G.parameters())/1e6:.1f} M | "
              f"D {sum(p.numel() for p in D.parameters())/1e6:.1f} M | native {H}x{W} | "
              f"global batch {global_batch} | gamma {args.r1_gamma}", flush=True)

    # gradients averaged by hand, as in stylegan2_lite (see the note there on DDP)
    if world > 1:
        for m in (G, D, G_ema):
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t.data, src=0)

    def allreduce_grads(module):
        if world == 1:
            return
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        flat = torch.cat([g.flatten() for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        torch.nan_to_num_(flat, nan=0, posinf=1e5, neginf=-1e5)   # as NVIDIA's loop does
        o = 0
        for g in grads:
            g.copy_(flat[o:o + g.numel()].view_as(g)); o += g.numel()

    # G has no lazy regulariser; D's R1 is lazy, folded into lr/betas (NVIDIA's mb_ratio)
    dr = args.d_reg_every / (args.d_reg_every + 1)
    optG = torch.optim.Adam(G.parameters(), lr=args.glr, betas=(0.0, 0.99), eps=1e-8)
    optD = torch.optim.Adam(D.parameters(), lr=args.dlr * dr, betas=(0.0, 0.99 ** dr), eps=1e-8)

    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch
    ckpt_p = os.path.join(args.out, "latest.pth")
    start_ep, gstep = 0, 0
    if os.path.exists(ckpt_p):
        c = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        G.load_state_dict(c["G"]); D.load_state_dict(c["D"]); G_ema.load_state_dict(c["G_ema"])
        optG.load_state_dict(c["optG"]); optD.load_state_dict(c["optD"])
        aug.p.copy_(torch.tensor(c["aug_p"]))
        start_ep, gstep = c["epoch"] + 1, c["gstep"]
        if main_proc: print(f"resumed ep {start_ep} step {gstep}", flush=True)

    ema_kimg, ema_rampup = global_batch * 10 / 32, 0.05
    t0, s0 = time.time(), gstep
    for ep in range(start_ep, args.epochs):
        if samp is not None: samp.set_epoch(ep)
        for real in dl:
            real = real.to(dev, non_blocking=True)
            n = real.shape[0]

            # ---------------- G main ----------------
            D.requires_grad_(False); G.requires_grad_(True)
            fake = G(torch.randn(n, G.z_dim, device=dev))
            loss_g = F.softplus(-D(aug(fake))).mean()
            optG.zero_grad(set_to_none=True); loss_g.backward(); allreduce_grads(G); optG.step()

            # ---------------- D main ----------------
            D.requires_grad_(True); G.requires_grad_(False)
            with torch.no_grad():
                fake = G(torch.randn(n, G.z_dim, device=dev), update_emas=True)
            d_fake = D(aug(fake))
            d_real = D(aug(real))
            loss_d = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
            optD.zero_grad(set_to_none=True); loss_d.backward(); allreduce_grads(D); optD.step()
            # ADA statistic over the global batch, not per rank
            sgn = torch.sign(d_real.detach()).mean()
            if world > 1:
                dist.all_reduce(sgn); sgn /= world
            p_now = ada.update(sgn.view(1))

            if gstep % args.d_reg_every == 0:               # lazy R1 (see stylegan2_lite)
                real_r = aug(real).detach().requires_grad_(True)
                d_r = D(real_r)
                grad = torch.autograd.grad(d_r.sum(), real_r, create_graph=True)[0]
                loss_r1 = (args.r1_gamma / 2) * grad.pow(2).sum([1, 2, 3]).mean() * args.d_reg_every
                optD.zero_grad(set_to_none=True); loss_r1.backward(); allreduce_grads(D); optD.step()

            # ---------------- G EMA (NVIDIA: ema_kimg, ramp-up 0.05) ----------------
            with torch.no_grad():
                cur_nimg = (gstep + 1) * global_batch
                ema_nimg = min(ema_kimg * 1000, cur_nimg * ema_rampup)
                beta = 0.5 ** (global_batch / max(ema_nimg, 1e-8))
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.copy_(pg.lerp(pe, beta))
                for be, bg in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(bg)

            gstep += 1
            if main_proc and gstep % 50 == 0:
                el = time.time() - t0
                rate = el / max(gstep - s0, 1)
                print(f"ep {ep} step {gstep}/{total_steps} d {loss_d.item():.3f} "
                      f"g {loss_g.item():.3f} aug_p {p_now:.3f} {rate:.2f}s/step {el/3600:.2f}h "
                      f"eta {rate*(total_steps-gstep)/3600:.1f}h", flush=True)
            if args.max_steps and gstep - s0 >= args.max_steps:
                if main_proc:
                    print(f"SMOKE done {args.max_steps} steps, {(time.time()-t0)/args.max_steps:.2f} s/step "
                          f"(incl. startup), peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
                if world > 1: dist.destroy_process_group()
                return

        if main_proc:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "aug_p": float(aug.p), "epoch": ep, "gstep": gstep}, ckpt_p + ".tmp")
            os.replace(ckpt_p + ".tmp", ckpt_p)
            json.dump({"epoch": ep, "gstep": gstep, "aug_p": float(aug.p),
                       "hours_this_run": (time.time()-t0)/3600, "global_batch": global_batch},
                      open(os.path.join(args.out, "progress.json"), "w"), indent=1)
        if world > 1: dist.barrier()
    if main_proc:
        print(f"DONE {args.epochs} epochs", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
