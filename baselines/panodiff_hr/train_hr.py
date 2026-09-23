"""Train PanoDiff directly at 1024x512 -- the control arm for reviewer R3.1.

Everything matches the released PanoDiff recipe (Project/sd/train.py) except the
pixel grid and the two extra UNet resolution levels that the grid forces:
  cosine beta schedule, T=1000, L1 loss on the predicted noise, AdamW
  lr 1e-4 betas (0.9, 0.99) wd 0, cosine LR schedule with 300 warmup steps,
  EMA with the cosine-ramped coefficient, fp16 autocast, seed 42, GLOBAL batch 4.
Global batch is held at 4 (PanoDiff's value) by running 4 ranks at batch 1, so
the optimiser sees an identical sequence of update sizes; only wall-clock differs.

DDP is srun-launched (SLURM_PROCID/SLURM_LOCALID), following the pattern already
used for the I-JEPA runs on LUMI.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

B = WORK + "/new-baselines-generation"
sys.path.insert(0, B)
sys.path.insert(0, WORK + "/project-files/PanoDiff/Project/sd")
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
from panodiff_hr.unet_hr import UNetHR                      # noqa: E402
from simple_diffusion.scheduler import DDIMScheduler        # noqa: E402
from simple_diffusion.ema import EMA                        # noqa: E402

SEED = 42


def setup_dist():
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        rank = int(os.environ["SLURM_PROCID"]); world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, local


class PRDataset(Dataset):
    """1024x512 radiographs, scaled to [-1, 1] exactly as in the released loader."""
    def __init__(self, listfile):
        self.files = [l.strip() for l in open(listfile) if l.strip()]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            im = im.convert("RGB")
            if im.size != (1024, 512):
                im = im.resize((1024, 512), Image.LANCZOS)
            x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
        return x.permute(2, 0, 1).float().div_(255.).mul_(2.).sub_(1.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--batch", type=int, default=1, help="PER-RANK batch; global must be 4")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.996)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    rank, world, local = setup_dist()
    is_main = rank == 0
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)

    ds = PRDataset(f"{B}/data/train_hr.txt")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=SEED) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    model = UNetHR().to(dev)
    if is_main:
        print(model.describe(), flush=True)
        print(f"global batch = {world*args.batch} (PanoDiff used 4)", flush=True)
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    ema = EMA(model, args.gamma, args.epochs * len(dl))
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local]) if world > 1 else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    total_steps = args.epochs * len(dl)
    def lr_at(s):
        if s < args.warmup: return s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    lrs = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    ckpt_path = os.path.join(args.out, "latest.pth")
    start_ep, gstep, gamma = 0, 0, args.gamma
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); ema.ema_model.load_state_dict(c["ema"])
        opt.load_state_dict(c["opt"]); lrs.load_state_dict(c["lrs"]); scaler.load_state_dict(c["scaler"])
        start_ep, gstep, gamma = c["epoch"] + 1, c["gstep"], c["gamma"]
        if is_main: print(f"resumed from epoch {start_ep} step {gstep}", flush=True)

    t_start = time.time()
    for ep in range(start_ep, args.epochs):
        if sampler is not None: sampler.set_epoch(ep)
        model.train(); run, n = 0.0, 0
        for x0 in dl:
            x0 = x0.to(dev, non_blocking=True)
            noise = torch.randn_like(x0)
            t = torch.randint(0, sched.num_train_timesteps, (x0.shape[0],), device=dev).long()
            xt = sched.add_noise(x0, noise, t)
            with torch.amp.autocast("cuda", enabled=True):
                loss = F.l1_loss(ddp(xt, t)["sample"], noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ema.update_params(gamma); gamma = ema.update_gamma(gstep)
            lrs.step(); gstep += 1
            run += loss.item(); n += 1
            if is_main and gstep % 200 == 0:
                el = time.time() - t_start
                print(f"ep {ep} step {gstep}/{total_steps} loss {run/n:.4f} "
                      f"lr {lrs.get_last_lr()[0]:.2e} {el/3600:.2f}h "
                      f"eta {el/gstep*(total_steps-gstep)/3600:.1f}h", flush=True)
        if is_main:
            torch.save({"model": model.state_dict(), "ema": ema.ema_model.state_dict(),
                        "opt": opt.state_dict(), "lrs": lrs.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": ep, "gstep": gstep, "gamma": gamma}, ckpt_path)
            json.dump({"epoch": ep, "gstep": gstep, "loss": run/max(n,1),
                       "hours": (time.time()-t_start)/3600, "global_batch": world*args.batch},
                      open(os.path.join(args.out, "progress.json"), "w"), indent=1)
    if is_main:
        print(f"DONE {args.epochs} epochs in {(time.time()-t_start)/3600:.2f} h", flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
