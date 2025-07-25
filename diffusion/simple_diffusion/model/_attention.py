import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange


class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1]**0.5)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, use_flash_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.use_flash_attn = use_flash_attn

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        self.norm = RMSNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        if not self.use_flash_attn:
            q = q * self.scale
            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            attn = sim.softmax(dim=-1)
            del sim
            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            del attn
            out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
            out = rearrange(out, 'b h d (x y) -> b (h d) x y', x=h, y=w)

        return self.to_out(out) + x
