"""FastGAN part of bench_train_cost.py, run separately."""

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


# ---------------------------------------------------------------- FastGAN --
os.chdir(FG); sys.path.insert(0, FG)
sys.argv = ["x"]
import lpips                                                   # noqa: E402  (vendored, LUMI-patched)
from models import Generator, Discriminator                    # noqa: E402
from diffaug import DiffAugment                                # noqa: E402


def crop_image_by_part(image, part):                           # FastGAN-pytorch/train.py
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]

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

prev = json.load(open(os.path.join(HERE, "train_cost.json"))) if os.path.exists(os.path.join(HERE, "train_cost.json")) else {}
prev.update(res)
json.dump(prev, open(os.path.join(HERE, "train_cost.json"), "w"), indent=2)
res = prev
print(json.dumps(res, indent=2))
