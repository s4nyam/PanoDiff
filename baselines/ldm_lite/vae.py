"""KL-regularised autoencoder for the latent-diffusion baseline.

Medfusion's own VAE diverged here: its latents reached +/-80,000 (std 6853) and
reconstruction MAE was 1.0 -- so its latent diffusion stage was learning on top
of a broken code, which is why those samples were pure noise. Rather than debug
that repo, this is the standard LDM autoencoder (Rombach et al., Sec. 3), which
is a small, well-understood model:

  * 3 stride-2 stages -> 8x spatial compression, 512x1024 -> 64x128
  * 4-channel latent (the Stable-Diffusion choice)
  * GroupNorm + SiLU residual blocks
  * L1 reconstruction + a very small KL term (1e-6), which is what keeps the
    latent scale bounded -- the exact failure mode seen above

Trained in bf16 with gradient clipping, after fp16 silently destabilised both
Medfusion stages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = nn.GroupNorm(32, cin); self.c1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2 = nn.GroupNorm(32, cout); self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, ch=64, mults=(1, 2, 4, 4), z_ch=4):
        super().__init__()
        self.inp = nn.Conv2d(3, ch, 3, 1, 1)
        blocks, cur = [], ch
        for i, m in enumerate(mults):
            out = ch * m
            blocks.append(ResBlock(cur, out)); cur = out
            if i < len(mults) - 1:                      # 3 downsamples = 8x
                blocks.append(nn.Conv2d(cur, cur, 3, 2, 1))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.GroupNorm(32, cur), nn.SiLU(),
                                 nn.Conv2d(cur, 2 * z_ch, 3, 1, 1))   # mu, logvar

    def forward(self, x):
        return self.out(self.blocks(self.inp(x))).chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self, ch=64, mults=(1, 2, 4, 4), z_ch=4):
        super().__init__()
        cur = ch * mults[-1]
        self.inp = nn.Conv2d(z_ch, cur, 3, 1, 1)
        blocks = []
        for i, m in enumerate(reversed(mults)):
            out = ch * m
            blocks.append(ResBlock(cur, out)); cur = out
            if i < len(mults) - 1:
                blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                blocks.append(nn.Conv2d(cur, cur, 3, 1, 1))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.GroupNorm(32, cur), nn.SiLU(),
                                 nn.Conv2d(cur, 3, 3, 1, 1))

    def forward(self, z):
        return self.out(self.blocks(self.inp(z)))


class VAE(nn.Module):
    def __init__(self, ch=64, mults=(1, 2, 4, 4), z_ch=4, kl_weight=1e-6):
        super().__init__()
        self.enc = Encoder(ch, mults, z_ch)
        self.dec = Decoder(ch, mults, z_ch)
        self.kl_weight = kl_weight
        self.z_ch = z_ch

    def encode(self, x, sample=True):
        mu, logvar = self.enc(x)
        logvar = logvar.clamp(-30, 20)                  # keeps sigma finite
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.enc(x)
        logvar = logvar.clamp(-30, 20)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = self.dec(z)
        rec_loss = F.l1_loss(rec, x)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean()
        return rec, rec_loss + self.kl_weight * kl, rec_loss.detach(), kl.detach()
