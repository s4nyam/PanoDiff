"""StyleGAN3-T port: (1) the fast filtered_lrelu equals NVIDIA's reference path on the
full generator, forward and backward; (2) time a GAN step on one GCD."""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import os, sys, time
import torch
import torch.nn.functional as F
B = WORK + "/new-baselines-generation"
sys.path.insert(0, B)
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
from stylegan3_lite import networks as N3          # noqa: E402
from stylegan2_lite.networks import Discriminator  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)
G = N3.Generator().to(dev)
print(f"G {sum(p.numel() for p in G.parameters())/1e6:.2f} M params, num_ws {G.num_ws}")
S = G.synthesis
print("input", list(S.input.size), "ch", S.input.channels, "sr", S.input.sampling_rate)
for n in S.layer_names:
    L = getattr(S, n)
    print(f"  {n:22s} in {list(L.in_size)} out {list(L.out_size)} sr {L.in_sampling_rate}->{L.out_sampling_rate} "
          f"up {L.up_factor} down {L.down_factor} taps {L.up_taps}/{L.down_taps} ch {L.in_channels}->{L.out_channels}")

# ---- (1) equivalence, full generator, fwd + bwd, batch 1
z = torch.randn(1, 512, device=dev)
def run(mode, up):
    N3.use_fast_op(mode == "fast"); N3._UP_MODE = up
    G.zero_grad(set_to_none=True)
    img = G(z)
    (img.square().mean() + img[:, :, ::7, ::5].mean()).backward()
    out = (img.detach().clone(), {k: p.grad.detach().clone() for k, p in G.named_parameters() if p.grad is not None})
    del img; torch.cuda.empty_cache()
    return out
o_r, g_r = run("ref", "zero")
print(f"ref out mean {o_r.mean().item():.5f} std {o_r.std().item():.5f} shape {tuple(o_r.shape)}")
for up in ["zero", "tconv"]:
    o_f, g_f = run("fast", up)
    worst = max(((g_r[k]-g_f[k]).abs().max().item() / (g_r[k].abs().max().item() + 1e-12), k) for k in g_r)
    print(f"EQUIV fast/{up}: output max|diff| {(o_r-o_f).abs().max().item():.3e} (scale {o_r.abs().max().item():.3e}); "
          f"grads {len(g_f)}/{len(g_r)} tensors, worst relative max|diff| {worst[0]:.3e} ({worst[1]})", flush=True)
del g_r, g_f; torch.cuda.empty_cache()
N3.use_fast_op(True)

# ---- (2) timing of one GAN step (D real+fake fwd/bwd, G fwd/bwd through D)
D = Discriminator(img_resolution=(512, 1024)).to(dev)
def step(bs):
    z = torch.randn(bs, 512, device=dev)
    with torch.no_grad():
        fake = G(z, update_emas=True)
    real = torch.randn(bs, 3, 512, 1024, device=dev)
    ld = F.softplus(D(fake)).mean() + F.softplus(-D(real)).mean()
    ld.backward()
    img = G(torch.randn(bs, 512, device=dev))
    lg = F.softplus(-D(img)).mean()
    lg.backward()

configs = [("fast", "group", "tconv"), ("fast", "reshape", "tconv"), ("fast", "group", "zero"), ("ref", "group", "zero")]
for mode, fir, up in configs:
    N3.use_fast_op(mode == "fast"); N3._FIR_MODE = fir; N3._UP_MODE = up
    for bs in ([1] if mode == "ref" else [1, 2]):
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3): step(bs)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): step(bs)
            torch.cuda.synchronize()
            dt = (time.time() - t) / 5
            print(f"TIME {mode:4s}/{fir:7s}/{up:5s} bs{bs}: {dt:.3f} s/step  {dt/bs:.3f} s/img  "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        except Exception as e:
            print(f"TIME {mode}/{fir}/{up} bs{bs}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()
