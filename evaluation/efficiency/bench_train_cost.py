"""Training cost of the two-stage pipelines in the accelerator-hours of Table 11.

Table 11 reports the direct high-resolution baselines in accelerator-hours on the
accelerators credited in the Acknowledgment, whereas Table 2 gives PanoDiff-SR's
training wall-clock on the single RTX 6000 Ada it was trained on. To put rows 1 and 2
on the same scale, this times one training step of each original stage on ONE of the
Table 11 accelerators, with the original batch sizes, and multiplies by the original
number of steps:

  PanoDiff   UNet 34.0 M, 128x256, batch 4, L1 on the noise, AdamW;
             110 epochs x ceil(7243/4) = 199,210 steps           (Table 2: 0.32 s/step)
  HAT-GAN    HAT 20.8 M on 64x64 -> 256x256 crops, batch 4; G: L1 + VGG19 perceptual
             + GAN (weight 0.1) with a UNet discriminator with spectral norm, EMA of G;
             D: real + fake; 400,000 iterations                   (Table 2: 0.48 s/step)
  FastGAN    im_size 512, batch 64, DiffAugment, LPIPS reconstruction loss in D;
             8,000 iterations (the last checkpoint of the recovered run, 2.4 h)

The HAT-GAN step omits the on-the-fly Real-ESRGAN degradation of the LR inputs (blur,
noise, JPEG), so its time is a lower bound; the ratio to the Table 2 time is printed
so the two can be compared.
"""
import json, os, sys, time, types, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SD = os.path.join(ROOT, "project-files/PanoDiff/Project/sd")
SR = os.path.join(ROOT, "project-files/PanoDiff/Project/sr/training")
FG = os.path.join(ROOT, "new-baselines-generation/repos/FastGAN-pytorch")
dev = torch.device("cuda")
N_WARM, N_TIME = 10, 40
res = {"gpu": torch.cuda.get_device_name(0)}


def timeit(step):
    for _ in range(N_WARM):
        step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_TIME):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_TIME


# --------------------------------------------------------------- PanoDiff --
sys.path.insert(0, SD)
from simple_diffusion.model import UNet                      # noqa: E402
unet = UNet(3, image_size=(128, 256), hidden_dims=[64, 128, 256, 512]).to(dev).train()
opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)


def pd_step():
    x = torch.randn(4, 3, 128, 256, device=dev)
    t = torch.randint(0, 1000, (4,), device=dev)
    loss = F.l1_loss(unet(x, t)["sample"], torch.randn_like(x))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


s = timeit(pd_step)
res["panodiff_s_per_step"] = s
res["panodiff_steps"] = 110 * ((7243 + 3) // 4)
res["panodiff_acc_h"] = s * res["panodiff_steps"] / 3600
print(f"PanoDiff  {s:.3f} s/step (Table 2: 0.32)  -> {res['panodiff_acc_h']:.1f} acc-h", flush=True)
del unet, opt; torch.cuda.empty_cache()

# ---------------------------------------------------------------- HAT-GAN --
reg = types.ModuleType("basicsr.utils.registry")
reg.ARCH_REGISTRY = types.SimpleNamespace(register=lambda *a, **k: (lambda c: c))
au = types.ModuleType("basicsr.archs.arch_util")
au.to_2tuple = lambda x: x if isinstance(x, tuple) else (x, x)
au.trunc_normal_ = torch.nn.init.trunc_normal_
au.scandir = lambda *a, **k: []
for name, mod in [("basicsr", types.ModuleType("basicsr")),
                  ("basicsr.utils", types.ModuleType("basicsr.utils")),
                  ("basicsr.archs", types.ModuleType("basicsr.archs")),
                  ("basicsr.utils.registry", reg), ("basicsr.archs.arch_util", au)]:
    sys.modules[name] = mod
sys.modules["basicsr.utils"].scandir = au.scandir
spec = importlib.util.spec_from_file_location("hat_arch", os.path.join(SR, "hat", "archs", "hat_arch.py"))
hm = importlib.util.module_from_spec(spec); spec.loader.exec_module(hm)


class UNetDiscriminatorSN(nn.Module):                      # basicsr.archs.discriminator_arch
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        n = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = n(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = n(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = n(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = n(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = n(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = n(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = n(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = n(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        lr = lambda t: F.leaky_relu(t, 0.2)
        x0 = lr(self.conv0(x)); x1 = lr(self.conv1(x0)); x2 = lr(self.conv2(x1)); x3 = lr(self.conv3(x2))
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = lr(self.conv4(x3)) + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = lr(self.conv5(x4)) + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = lr(self.conv6(x5)) + x0
        return self.conv9(lr(self.conv8(lr(self.conv7(x6)))))


import torchvision                                            # noqa: E402
vgg = torchvision.models.vgg19(weights=None).features[:36].to(dev).eval().requires_grad_(False)
LAYERS = {3: 0.1, 8: 0.1, 17: 1.0, 26: 1.0, 35: 1.0}          # conv1_2 conv2_2 conv3_4 conv4_4 conv5_4 (post-ReLU idx-1)


def vgg_feats(x):
    out, h = [], x
    for i, m in enumerate(vgg):
        h = m(h)
        if i in LAYERS:
            out.append((LAYERS[i], h))
    return out


G = hm.HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30,
           conv_scale=0.01, overlap_ratio=0.5, img_range=1., depths=[6] * 6, embed_dim=180,
           num_heads=[6] * 6, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv").to(dev).train()
G_ema = hm.HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30,
               conv_scale=0.01, overlap_ratio=0.5, img_range=1., depths=[6] * 6, embed_dim=180,
               num_heads=[6] * 6, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv").to(dev).eval()
D = UNetDiscriminatorSN().to(dev).train()
oG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.99))
oD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.99))


def hat_step():
    lq = torch.rand(4, 3, 64, 64, device=dev); gt = torch.rand(4, 3, 256, 256, device=dev)
    D.requires_grad_(False)
    out = G(lq)
    l_pix = F.l1_loss(out, gt)
    fo, fg = vgg_feats(out), vgg_feats(gt)
    l_per = sum(w * F.l1_loss(a, b.detach()) for (w, a), (_, b) in zip(fo, fg))
    l_gan = 0.1 * F.binary_cross_entropy_with_logits(D(out), torch.ones(4, 1, 256, 256, device=dev))
    oG.zero_grad(set_to_none=True); (l_pix + l_per + l_gan).backward(); oG.step()
    D.requires_grad_(True)
    lr_ = F.binary_cross_entropy_with_logits(D(gt), torch.ones(4, 1, 256, 256, device=dev))
    lf = F.binary_cross_entropy_with_logits(D(out.detach()), torch.zeros(4, 1, 256, 256, device=dev))
    oD.zero_grad(set_to_none=True); (lr_ + lf).backward(); oD.step()
    with torch.no_grad():
        for pe, pg in zip(G_ema.parameters(), G.parameters()):
            pe.mul_(0.999).add_(pg, alpha=0.001)


s = timeit(hat_step)
res["hat_s_per_step"] = s
res["hat_steps"] = 400000
res["hat_acc_h"] = s * 400000 / 3600
print(f"HAT-GAN   {s:.3f} s/step (Table 2: 0.48)  -> {res['hat_acc_h']:.1f} acc-h", flush=True)
del G, G_ema, D, oG, oD, vgg; torch.cuda.empty_cache()

# ---------------------------------------------------------------- FastGAN --
os.chdir(FG); sys.path.insert(0, FG)
sys.argv = ["x"]
import lpips                                                   # noqa: E402  (vendored, LUMI-patched)
from models import Generator, Discriminator                    # noqa: E402
from diffaug import DiffAugment                                # noqa: E402
from operation import crop_image_by_part                       # noqa: E402
percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
IM, BS, NZ = 512, 64, 256
netG = Generator(ngf=64, nz=NZ, im_size=IM).to(dev)
netD = Discriminator(ndf=64, im_size=IM).to(dev)
optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
policy = "color,translation"


def train_d(data, label):
    if label == "real":
        part = torch.randint(0, 4, (1,)).item()
        pred, [rec_all, rec_small, rec_part] = netD(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
            percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
            percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
            percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
    else:
        pred = netD(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()


def fg_step():
    real = torch.rand(BS, 3, IM, IM, device=dev) * 2 - 1
    noise = torch.randn(BS, NZ, device=dev)
    fake = netG(noise)
    real = DiffAugment(real, policy=policy)
    fake = [DiffAugment(f, policy=policy) for f in fake]
    netD.zero_grad()
    train_d(real, "real")
    train_d([f.detach() for f in fake], "fake")
    optD.step()
    netG.zero_grad()
    pred_g = netD(fake, "fake")
    (-pred_g.mean()).backward()
    optG.step()


s = timeit(fg_step)
res["fastgan_s_per_step"] = s
res["fastgan_steps"] = 8000
res["fastgan_acc_h"] = s * 8000 / 3600
print(f"FastGAN   {s:.3f} s/step (recovered run: 2.4 h / 8000 = 1.08 s)  -> {res['fastgan_acc_h']:.1f} acc-h", flush=True)

json.dump(res, open(os.path.join(HERE, "train_cost.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
