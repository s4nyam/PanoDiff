"""FastGAN (Liu et al., ICLR 2021) in plain PyTorch.

The paper's own repo trained here for 8 h without diverging and still produced
pure noise -- data loading, normalisation and the loss curves were all fine, so
it was a hyper-parameter/implementation issue not worth guessing at. This is a
direct rebuild of the two ideas that define the method:

  1. Skip-Layer Excitation (SLE, Sec. 3.1): a channel-wise gate computed from a
     LOW-resolution feature map and applied to a map 16x larger. It gives the
     generator a cheap long-range shortcut, which is what lets FastGAN train at
     1024 on very little data.
  2. A self-supervised discriminator (Sec. 3.2): small decoders hang off two
     intermediate D features and must reconstruct the real image. That
     reconstruction loss regularises D and is the paper's main anti-overfitting
     device, playing the role ADA plays in StyleGAN2-ADA.

Trained with the hinge loss, as in the paper.

Two deliberate departures, both stated in the response text:
  * NATIVE 512x1024. The reference implementation resizes to a square; writing
    it here allows a 4x8 base, so the radiographs keep their aspect ratio.
  * The decoder reconstruction target uses L1 rather than a VGG perceptual loss,
    removing an ImageNet-pretrained dependency -- the same pre-training confound
    R4 Maj-2 already criticises elsewhere in this paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_glu(cin, cout, k=3, s=1, p=1):
    """GLU halves the channel count, so the conv emits 2*cout."""
    return nn.Sequential(nn.Conv2d(cin, cout * 2, k, s, p, bias=False),
                         nn.BatchNorm2d(cout * 2), nn.GLU(dim=1))


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = conv_bn_glu(cin, cout)

    def forward(self, x):
        return self.block(self.up(x))


class SLE(nn.Module):
    """Skip-Layer Excitation: gate a high-res map with a low-res one."""
    def __init__(self, c_low, c_high):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(c_low, c_high, 4, 1, 0, bias=False), nn.SiLU(),
            nn.Conv2d(c_high, c_high, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, low, high):
        return high * self.main(low)


# channel multipliers keyed by the SHORT side, as in the reference config
NFC = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25}


class Generator(nn.Module):
    def __init__(self, nz=256, ngf=64, out_hw=(512, 1024)):
        super().__init__()
        self.nz = nz
        ch = {k: max(8, int(ngf * m)) for k, m in NFC.items()}
        self.init = nn.Sequential(                      # z -> 4x8
            nn.ConvTranspose2d(nz, ch[4] * 2, (4, 8), 1, 0, bias=False),
            nn.BatchNorm2d(ch[4] * 2), nn.GLU(dim=1))
        self.u8   = UpBlock(ch[4],   ch[8])
        self.u16  = UpBlock(ch[8],   ch[16])
        self.u32  = UpBlock(ch[16],  ch[32])
        self.u64  = UpBlock(ch[32],  ch[64])
        self.u128 = UpBlock(ch[64],  ch[128])
        self.u256 = UpBlock(ch[128], ch[256])
        self.u512 = UpBlock(ch[256], ch[512])
        # the 16x-jump skip connections of the paper
        self.se64  = SLE(ch[4],  ch[64])
        self.se128 = SLE(ch[8],  ch[128])
        self.se256 = SLE(ch[16], ch[256])
        self.se512 = SLE(ch[32], ch[512])
        self.to_rgb = nn.Sequential(nn.Conv2d(ch[512], 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z):
        f4   = self.init(z.view(z.size(0), self.nz, 1, 1))
        f8   = self.u8(f4)
        f16  = self.u16(f8)
        f32  = self.u32(f16)
        f64  = self.se64(f4,   self.u64(f32))
        f128 = self.se128(f8,  self.u128(f64))
        f256 = self.se256(f16, self.u256(f128))
        f512 = self.se512(f32, self.u512(f256))
        return self.to_rgb(f512)


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


class SimpleDecoder(nn.Module):
    """Reconstructs a small image from a D feature map (the self-supervision)."""
    def __init__(self, cin, cout=3):
        super().__init__()
        c = cin
        layers = []
        for _ in range(4):                              # 4 upsamples -> 16x
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       conv_bn_glu(c, max(8, c // 2))]
            c = max(8, c // 2)
        layers += [nn.Conv2d(c, cout, 3, 1, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, in_hw=(512, 1024)):
        super().__init__()
        ch = {k: max(8, int(ndf * m)) for k, m in NFC.items()}
        self.from_rgb = nn.Sequential(nn.Conv2d(3, ch[512], 3, 1, 1, bias=False),
                                      nn.LeakyReLU(0.2, inplace=True))
        self.d256 = down(ch[512], ch[256])
        self.d128 = down(ch[256], ch[128])
        self.d64  = down(ch[128], ch[64])
        self.d32  = down(ch[64],  ch[32])
        self.d16  = down(ch[32],  ch[16])
        self.d8   = down(ch[16],  ch[8])
        # six stride-2 stages take 512x1024 down to 8x16, so the final kernel is
        # (8,16) -- a (4,8) kernel would leave a 5x9 map and emit 45 logits.
        self.out  = nn.Conv2d(ch[8], 1, (8, 16), 1, 0, bias=False)
        self.dec16 = SimpleDecoder(ch[16])              # self-supervised heads
        self.dec8  = SimpleDecoder(ch[8])

    def forward(self, x, recon=False):
        h = self.from_rgb(x)
        h = self.d256(h); h = self.d128(h); h = self.d64(h); h = self.d32(h)
        f16 = self.d16(h)
        f8 = self.d8(f16)
        logit = self.out(f8).flatten(1)
        if not recon:
            return logit
        return logit, self.dec16(f16), self.dec8(f8)
