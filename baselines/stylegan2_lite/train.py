"""Train the reimplemented StyleGAN2-ADA on the panoramic radiographs.

Training recipe follows the paper: non-saturating logistic loss, lazy R1
regularisation on the discriminator every 16 steps, lazy path-length
regularisation on the generator every 4 steps, style-mixing regularisation with
probability 0.9, Adam(0, 0.99), and an EMA of the generator weights used for all
sampling. ADA holds the discriminator's r_t at 0.6.

Budget is the shared matched-epoch one: 110 epochs x 7243 images.
Native 512x1024 -- no squash-to-square, unlike NVIDIA's square-only release.
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
from stylegan2_lite.networks import Generator, Discriminator      # noqa: E402
from stylegan2_lite.augment import AugmentPipe, ADAController     # noqa: E402

SEED = 42
H, W = 512, 1024


def setup_dist():
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--batch", type=int, default=4, help="per rank")
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--r1-gamma", type=float, default=8.2)
    ap.add_argument("--d-reg-every", type=int, default=16)
    ap.add_argument("--g-reg-every", type=int, default=4)
    ap.add_argument("--ema-kimg", type=float, default=10.0)
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

    G = Generator(img_resolution=(H, W)).to(dev)
    D = Discriminator(img_resolution=(H, W)).to(dev)
    G_ema = Generator(img_resolution=(H, W)).to(dev).eval().requires_grad_(False)
    G_ema.load_state_dict(G.state_dict())
    aug = AugmentPipe().to(dev)
    global_batch = world * args.batch
    ada = ADAController(aug, target=0.6, interval=4, batch=global_batch)

    if main_proc:
        print(f"G {sum(p.numel() for p in G.parameters())/1e6:.1f} M | "
              f"D {sum(p.numel() for p in D.parameters())/1e6:.1f} M | "
              f"native {H}x{W} | global batch {global_batch}", flush=True)

    # No DDP wrappers. A GAN step runs D twice (real, fake) before one backward and
    # runs D again inside the G step with its parameters frozen; DDP's one-forward-
    # per-backward bucket accounting rejects both patterns ("Expected to have
    # finished reduction in the prior iteration"). Concatenating real and fake into
    # a single D forward would satisfy DDP but corrupt the minibatch-stddev layer,
    # which must see the two sets separately. So gradients are averaged by hand.
    if world > 1:
        for m in (G, D, G_ema):
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t.data, src=0)

    def allreduce_grads(module):
        if world == 1:
            return
        for prm in module.parameters():
            if prm.grad is not None:
                dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
                prm.grad.div_(world)

    Gd, Dd = G, D

    # lazy regularisation: fold the extra step into the LR/betas (paper Appendix B)
    gr, dr = args.g_reg_every / (args.g_reg_every + 1), args.d_reg_every / (args.d_reg_every + 1)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr * gr, betas=(0.0, 0.99 ** gr))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr * dr, betas=(0.0, 0.99 ** dr))

    total_steps = args.epochs * len(dl)
    ckpt_p = os.path.join(args.out, "latest.pth")
    start_ep, gstep, pl_mean = 0, 0, torch.zeros([], device=dev)
    if os.path.exists(ckpt_p):
        c = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        G.load_state_dict(c["G"]); D.load_state_dict(c["D"]); G_ema.load_state_dict(c["G_ema"])
        optG.load_state_dict(c["optG"]); optD.load_state_dict(c["optD"])
        aug.p.copy_(torch.tensor(c["aug_p"])); pl_mean = torch.tensor(c["pl_mean"], device=dev)
        start_ep, gstep = c["epoch"] + 1, c["gstep"]
        if main_proc: print(f"resumed ep {start_ep} step {gstep}", flush=True)

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        if samp is not None: samp.set_epoch(ep)
        for real in dl:
            real = real.to(dev, non_blocking=True)
            n = real.shape[0]

            # ---------------- D step ----------------
            D.requires_grad_(True); G.requires_grad_(False)
            z = torch.randn(n, G.z_dim, device=dev)
            with torch.no_grad():
                fake = Gd(z)
            d_fake = Dd(aug(fake))
            d_real = Dd(aug(real))
            loss_d = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
            optD.zero_grad(set_to_none=True); loss_d.backward(); allreduce_grads(D); optD.step()
            p_now = ada.update(d_real)

            if gstep % args.d_reg_every == 0:               # lazy R1
                # The penalty needs a double backward. Differentiating through the
                # ADA pipeline would require a second derivative of grid_sample,
                # which PyTorch does not implement (NVIDIA ships a custom
                # grid_sample_gradfix for exactly this). Augmenting first and
                # taking the gradient w.r.t. the augmented image keeps R1 doing its
                # job -- penalising D's sensitivity to its own input -- while the
                # double backward stays inside D, where it is supported.
                real_r = aug(real).detach().requires_grad_(True)
                d_r = Dd(real_r)
                grad = torch.autograd.grad(d_r.sum(), real_r, create_graph=True)[0]
                r1 = grad.pow(2).sum([1, 2, 3]).mean()
                loss_r1 = (args.r1_gamma / 2) * r1 * args.d_reg_every
                optD.zero_grad(set_to_none=True); loss_r1.backward(); allreduce_grads(D); optD.step()

            # ---------------- G step ----------------
            D.requires_grad_(False); G.requires_grad_(True)
            z = torch.randn(n, G.z_dim, device=dev)
            ws = G.mapping(z)
            if torch.rand(()) < 0.9:                        # style mixing
                z2 = torch.randn(n, G.z_dim, device=dev)
                ws2 = G.mapping(z2, update_w_avg=False)
                cut = int(torch.randint(1, ws.shape[1], ()))
                ws = torch.cat([ws[:, :cut], ws2[:, cut:]], 1)
            fake = Gd(None, ws=ws)
            loss_g = F.softplus(-Dd(aug(fake))).mean()
            optG.zero_grad(set_to_none=True); loss_g.backward(); allreduce_grads(G); optG.step()

            if gstep % args.g_reg_every == 0:               # lazy path length
                nh = max(n // 2, 1)
                zp = torch.randn(nh, G.z_dim, device=dev)
                wp = G.mapping(zp, update_w_avg=False).requires_grad_(True)
                imp = Gd(None, ws=wp)
                noise = torch.randn_like(imp) / math.sqrt(imp.shape[2] * imp.shape[3])
                gr_ = torch.autograd.grad((imp * noise).sum(), wp, create_graph=True)[0]
                pl = gr_.pow(2).sum(2).mean(1).sqrt()
                pl_mean = pl_mean.lerp(pl.mean().detach(), 0.01)
                loss_pl = (pl - pl_mean).pow(2).mean() * 2.0 * args.g_reg_every
                optG.zero_grad(set_to_none=True); loss_pl.backward(); allreduce_grads(G); optG.step()

            with torch.no_grad():                          # EMA of G
                beta = 0.5 ** (global_batch / (args.ema_kimg * 1000))
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.copy_(pg.lerp(pe, beta))
                for be, bg in zip(G_ema.buffers(), G.buffers()):
                    be.copy_(bg)

            gstep += 1
            if main_proc and gstep % 100 == 0:
                el = time.time() - t0
                print(f"ep {ep} step {gstep}/{total_steps} d {loss_d.item():.3f} "
                      f"g {loss_g.item():.3f} aug_p {p_now:.3f} {el/3600:.2f}h "
                      f"eta {el/max(gstep-1,1)*(total_steps-gstep)/3600:.1f}h", flush=True)

        if main_proc:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "aug_p": float(aug.p), "pl_mean": float(pl_mean),
                        "epoch": ep, "gstep": gstep}, ckpt_p)
            json.dump({"epoch": ep, "gstep": gstep, "aug_p": float(aug.p),
                       "hours": (time.time()-t0)/3600, "global_batch": global_batch},
                      open(os.path.join(args.out, "progress.json"), "w"), indent=1)
    if main_proc:
        print(f"DONE {args.epochs} epochs in {(time.time()-t0)/3600:.2f} h", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
