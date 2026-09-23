"""Emit 7243 synthetic PRs at 1024x512 PNG from a trained baseline.

Everything downstream depends on these folders being directly comparable, so
every model writes the SAME count at the SAME final resolution in the SAME
format, and the FID/IS scorer then treats them identically.

Sampling is embarrassingly parallel -- unlike training there is no global-batch
constraint -- so the 7243 images are simply sharded across all ranks by index
and each rank writes its own disjoint slice. Seeds are derived from the global
image index, so the output is reproducible and independent of the rank layout.

Square-only models (ADM, FastGAN) generate 1024x1024 and are resized back to
1024x512 here, inverting the anisotropic squash used to train them. Models that
trained natively at 512x1024 (PanoDiff-HR, StyleGAN2-ADA, Medfusion) are written
as-is.
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

B = WORK + "/new-baselines-generation"
sys.path.insert(0, B)
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
N_TARGET, OUT_W, OUT_H = 7243, 1024, 512


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


def save_batch(arr, idxs, outdir):
    """arr: float [-1,1] tensor [N,3,H,W] -> 1024x512 uint8 PNG."""
    arr = ((arr.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    for a, i in zip(arr, idxs):
        im = Image.fromarray(a)
        if im.size != (OUT_W, OUT_H):
            im = im.resize((OUT_W, OUT_H), Image.LANCZOS)
        im.save(os.path.join(outdir, f"gen_{i:06d}.png"), compress_level=1)


# ------------------------------------------------------------------ builders
def build_panodiff_hr(dev):
    sys.path.insert(0, WORK + "/"
                       "project-files/PanoDiff/Project/sd")
    from panodiff_hr.unet_hr import UNetHR
    from simple_diffusion.scheduler import DDIMScheduler
    net = UNetHR().to(dev).eval()
    c = torch.load(f"{B}/runs/panodiff_hr/latest.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(c["ema"])                     # EMA weights, as the paper samples
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    def gen(n, seed):
        g = torch.Generator().manual_seed(seed)
        # generate() defaults device to a hardcoded 'cuda:0'; pass this rank's
        # device explicitly or rank>0 mixes cuda:0 noise with a cuda:N model.
        out = sched.generate(net, num_inference_steps=250, generator=g, eta=1.0,
                             batch_size=n, device=dev)
        return out["sample_pt"].to(dev) * 2 - 1      # generate() returns [0,1]
    return gen


def build_sg2ada(dev):
    from stylegan2_lite.networks import Generator
    G = Generator(img_resolution=(OUT_H, OUT_W)).to(dev).eval()
    c = torch.load(f"{B}/runs/sg2ada_lite/latest.pth", map_location="cpu", weights_only=False)
    G.load_state_dict(c["G_ema"])
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return G(torch.randn(n, G.z_dim, device=dev), truncation_psi=1.0)
    return gen


def build_sg3(dev):
    from stylegan3_lite.networks import Generator
    G = Generator(img_resolution=(OUT_H, OUT_W)).to(dev).eval()
    c = torch.load(f"{B}/runs/sg3_lite/latest.pth", map_location="cpu", weights_only=False)
    G.load_state_dict(c["G_ema"])
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return G(torch.randn(n, G.z_dim, device=dev), truncation_psi=1.0)
    return gen


def build_fastgan(dev):
    sys.path.insert(0, f"{B}/repos/FastGAN-pytorch")
    from models import Generator
    from operation import load_params
    G = Generator(ngf=64, nz=256, im_size=1024).to(dev).eval()
    c = torch.load(f"{B}/runs/fastgan_hr/train_results/fastgan_hr/models/all_99591.pth",
                   map_location="cpu", weights_only=False)
    load_params(G, c["g_ema"])                        # g_ema is a param LIST, not a state_dict
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        out = G(torch.randn(n, 256, device=dev))
        return out[0] if isinstance(out, (list, tuple)) else out
    return gen


def build_adm(dev):
    sys.path.insert(0, f"{B}/repos/guided-diffusion")
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    d = model_and_diffusion_defaults()
    d.update(image_size=1024, num_channels=128, num_res_blocks=2,
             attention_resolutions="32,16,8", learn_sigma=True, class_cond=False,
             diffusion_steps=1000, noise_schedule="cosine", use_fp16=True,
             resblock_updown=True, use_scale_shift_norm=True, timestep_respacing="ddim250")
    model, diffusion = create_model_and_diffusion(**d)
    model.load_state_dict(torch.load(f"{B}/runs/adm/model024898.pt", map_location="cpu"))
    model.to(dev).eval(); model.convert_to_fp16()
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return diffusion.ddim_sample_loop(model, (n, 3, 1024, 1024),
                                          clip_denoised=True, progress=False)
    return gen


def build_medfusion(dev):
    sys.path.insert(0, f"{B}/repos/medfusion")
    sys.path.insert(0, f"{B}/medfusion_run")
    from train_medfusion import build_ldm
    pipe = build_ldm(f"{B}/runs/medfusion_vae/last.ckpt")
    c = torch.load(f"{B}/runs/medfusion_ldm/last.ckpt", map_location="cpu", weights_only=False)
    pipe.load_state_dict(c["state_dict"], strict=False)
    pipe = pipe.to(dev).eval()
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return pipe.sample(n, (8, OUT_H // 8, OUT_W // 8))   # 8-ch latent, 8x compression
    return gen


def build_fastgan_lite(dev):
    """Our FastGAN rebuild -- native 512x1024, EMA generator."""
    from fastgan_lite.networks import Generator
    G = Generator(nz=256, out_hw=(OUT_H, OUT_W)).to(dev).eval()
    c = torch.load(f"{B}/runs/fastgan_lite/latest.pth", map_location="cpu", weights_only=False)
    G.load_state_dict(c["G_ema"])
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return G(torch.randn(n, 256, device=dev))
    return gen


def build_fastgan_lite64(dev):
    """The same rebuild retrained at global batch 64, the batch size the paper's own
    successful FastGAN run used (recovered args.txt: im_size=512, batch_size=64).
    The first attempt used global batch 32 at 1024x512 and mode-collapsed."""
    from fastgan_lite.networks import Generator
    G = Generator(nz=256, out_hw=(OUT_H, OUT_W)).to(dev).eval()
    c = torch.load(f"{B}/runs/fastgan_lite64/latest.pth", map_location="cpu", weights_only=False)
    G.load_state_dict(c["G_ema"])
    @torch.no_grad()
    def gen(n, seed):
        torch.manual_seed(seed)
        return G(torch.randn(n, 256, device=dev))
    return gen


def build_ldm(dev):
    """Our latent diffusion -- DDIM in the VAE latent space, then decode."""
    sys.path.insert(0, WORK + "/"
                       "project-files/PanoDiff/Project/sd")
    from ldm_lite.vae import VAE
    from simple_diffusion.model.unet import UNet
    from simple_diffusion.scheduler import DDIMScheduler
    cv = torch.load(f"{B}/runs/ldm_vae/latest.pth", map_location="cpu", weights_only=False)
    vae = VAE(z_ch=4).to(dev).eval(); vae.load_state_dict(cv["model"]); zs = cv["latent_scale"]
    net = UNet(4, image_size=(OUT_H // 8, OUT_W // 8), hidden_dims=[128, 256, 384, 512],
               use_flash_attn=False).to(dev)
    net.conv_out = torch.nn.Conv2d(128, 4, 1).to(dev)
    cl = torch.load(f"{B}/runs/ldm_ldm/latest.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(cl["model"]); net.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    @torch.no_grad()
    def gen(n, seed):
        g = torch.Generator().manual_seed(seed)
        z = sched.generate(net, num_inference_steps=250, generator=g, eta=1.0,
                           batch_size=n, device=dev)["sample_pt"]
        return vae.decode((z * 2 - 1) / zs)
    return gen


BUILDERS = dict(panodiff_hr=build_panodiff_hr, sg2ada=build_sg2ada, sg3=build_sg3, fastgan_hr=build_fastgan,
                adm=build_adm, medfusion=build_medfusion,
                fastgan_lite=build_fastgan_lite, ldm=build_ldm,
                fastgan_lite64=build_fastgan_lite64)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BUILDERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    rank, world, local = setup()
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)

    idxs = list(range(rank, N_TARGET, world))          # disjoint shard per rank
    gen = BUILDERS[a.model](dev)
    if rank == 0:
        print(f"{a.model}: {N_TARGET} images over {world} ranks "
              f"(~{len(idxs)}/rank) -> {a.out}", flush=True)
    t0 = time.time(); done = 0
    for s in range(0, len(idxs), a.batch):
        chunk = idxs[s:s + a.batch]
        todo = [i for i in chunk if not os.path.exists(os.path.join(a.out, f"gen_{i:06d}.png"))]
        if todo:
            imgs = gen(len(todo), seed=1000 + todo[0])
            save_batch(imgs, todo, a.out)
        done += len(chunk)
        if rank == 0 and done % (a.batch * 10) == 0:
            el = time.time() - t0
            print(f"  {done}/{len(idxs)} per rank | {el/60:.1f} min | "
                  f"eta {el/max(done,1)*(len(idxs)-done)/60:.1f} min", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        n = len([f for f in os.listdir(a.out) if f.endswith('.png')])
        print(f"DONE {a.model}: {n} images in {a.out} ({(time.time()-t0)/60:.1f} min)", flush=True)
    if world > 1:
        dist.destroy_process_group()
