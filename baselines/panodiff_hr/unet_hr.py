"""PanoDiff UNet scaled to native 1024x512 synthesis (baseline #1, reviewer R3.1).

This is the two-stage design's control arm: the SAME denoiser, data, schedule and
sampler as PanoDiff, generating the full-resolution radiograph directly instead of
a 256x128 seed that is then super-resolved. Only the pixel grid differs, so the
comparison isolates the two-stage arrangement itself.

Scaling rule.  The released model uses hidden_dims=[64,128,256,512] on a 128x256
input: three down-blocks, two pixel-unshuffle halvings (the last block does not
reduce), coarsest map 32x64, and self-attention at the two coarsest levels --
64x128 (8192 tokens) and 32x64 (2048 tokens).  Naively raising the input to
512x1024 with that config would put attention on a 256x512 map (131k tokens, a
1.7e10-entry score matrix) and OOM immediately.

So we add two resolution levels rather than widening the existing ones, and gate
attention on the SPATIAL SIZE instead of the block index.  Attention therefore
lands on exactly the same 64x128 and 32x64 maps as in the released model, and the
two extra levels are pure convolutional resolution stages.  That keeps the
attention budget, the receptive field at the coarsest scale, and the bottleneck
resolution identical between the two arms.

    input 512x1024 -> 256x512 -> 128x256 -> 64x128 -> 32x64 (bottleneck)
    attention:  off      off       off       ON       ON
"""
import os as _os
WORK = _os.environ.get("PANODIFF_WORK", _os.path.abspath("work"))  # work directory, see docs/REPRODUCE.md
import sys
import torch
import torch.nn as nn

sys.path.insert(0, WORK + "/"
                   "project-files/PanoDiff/Project/sd")
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # this repository
sys.path[:0] = [_os.path.join(_REPO, "baselines"), _os.path.join(_REPO, "panodiff")]
from simple_diffusion.model.unet import (  # noqa: E402
    ResidualBlock, Attention, sinusoidal_embedding,
    get_downsample_layer, get_upsample_layer, get_attn_layer)

# Attention is applied only where the feature map has at most this many tokens.
# 64*128 = 8192 is the finest attended map in the released 128x256 model.
ATTN_MAX_TOKENS = 8192


class UNetHR(nn.Module):
    def __init__(self, in_channels=3,
                 hidden_dims=(64, 128, 256, 384, 512, 512),
                 image_size=(512, 1024),
                 use_flash_attn=False):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.sample_size = tuple(image_size)
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims

        timestep_input_dim = hidden_dims[0]
        time_embed_dim = timestep_input_dim * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim))
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, 1, 1)

        H, W = image_size
        self.attn_plan = {"down": [], "up": []}

        # ---- encoder ---------------------------------------------------------
        down_blocks = []
        in_dim = hidden_dims[0]
        h, w = H, W
        for idx, hidden_dim in enumerate(hidden_dims[1:]):
            is_last = idx >= (len(hidden_dims) - 2)
            use_attn = (h * w) <= ATTN_MAX_TOKENS
            self.attn_plan["down"].append((f"{h}x{w}", use_attn))
            down_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim, in_dim, time_embed_dim),
                ResidualBlock(in_dim, in_dim, time_embed_dim),
                get_attn_layer(in_dim, use_attn, use_flash_attn),
                get_downsample_layer(in_dim, hidden_dim, is_last)]))
            in_dim = hidden_dim
            if not is_last:
                h, w = h // 2, w // 2
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottleneck_hw = (h, w)

        mid_dim = hidden_dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim)

        # ---- decoder (mirror of the encoder) ---------------------------------
        up_blocks = []
        in_dim = mid_dim
        for idx, hidden_dim in enumerate(list(reversed(hidden_dims[:-1]))):
            is_last = idx >= (len(hidden_dims) - 2)
            use_attn = (h * w) <= ATTN_MAX_TOKENS
            self.attn_plan["up"].append((f"{h}x{w}", use_attn))
            up_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                get_attn_layer(in_dim, use_attn, use_flash_attn),
                get_upsample_layer(in_dim, hidden_dim, is_last)]))
            in_dim = hidden_dim
            if not is_last:
                h, w = h * 2, w * 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out_block = ResidualBlock(hidden_dims[0] * 2, hidden_dims[0], time_embed_dim)
        self.conv_out = nn.Conv2d(hidden_dims[0], 3, kernel_size=1)

    def describe(self):
        lines = [f"UNetHR  input {self.sample_size}  bottleneck {self.bottleneck_hw}",
                 f"  hidden_dims {self.hidden_dims}",
                 f"  params {sum(p.numel() for p in self.parameters())/1e6:.1f} M"]
        for side in ("down", "up"):
            for res, on in self.attn_plan[side]:
                lines.append(f"  {side:4s} {res:>9s}  attn={'ON ' if on else 'off'}")
        return "\n".join(lines)

    def forward(self, sample, timesteps):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        timesteps = torch.flatten(timesteps).broadcast_to(sample.shape[0])
        t_emb = self.time_embedding(sinusoidal_embedding(timesteps, self.hidden_dims[0]))

        x = self.init_conv(sample)
        r = x.clone()
        skips = []
        for block1, block2, attn, downsample in self.down_blocks:
            x = block1(x, t_emb); skips.append(x)
            x = block2(x, t_emb); x = attn(x); skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t_emb); x = self.mid_attn(x); x = self.mid_block2(x, t_emb)
        for block1, block2, attn, upsample in self.up_blocks:
            x = torch.cat((x, skips.pop()), dim=1); x = block1(x, t_emb)
            x = torch.cat((x, skips.pop()), dim=1); x = block2(x, t_emb); x = attn(x)
            x = upsample(x)
        x = self.out_block(torch.cat((x, r), dim=1), t_emb)
        return {"sample": self.conv_out(x)}
