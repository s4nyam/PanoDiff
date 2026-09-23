# Direct high-resolution baselines (paper Section 4.5.2, Table 11)

Every model generates 1024 × 512 radiographs directly and was trained from scratch on the same
7243 real PRs for 110 epochs (the budget of PanoDiff), with each method's own recommended
settings. All StyleGAN, FastGAN and LDM models train natively at 512 × 1024 (height × width); only
ADM, whose code is square-only, trained on 1024 × 1024 squashed images that were resized back.

| Folder | Model | Paper row | Training command (per rank) | Ranks | Cost (acc.-h) |
|---|---|---|---|---|---|
| `stylegan2_lite/` | StyleGAN2-ADA (Karras et al., 2020) | 3 | `python train.py --out runs/sg2ada --epochs 110 --batch 1 --r1-gamma 8.2` | 32 | 441 |
| `stylegan3_lite/` | StyleGAN3-T (Karras et al., 2021) | 4 | `python train.py --out runs/sg3 --epochs 110 --batch 1 --r1-gamma 8.2` | 32 | 572 |
| `fastgan_lite/` | FastGAN (Liu et al., 2021), direct HR | 5 | `python train.py --out runs/fastgan_hr --epochs 110 --batch 2` | 32 | 11 |
| `adm/` | ADM (Dhariwal & Nichol, 2021) | 6 | see `adm/README.md` | 32 | 122 |
| `ldm_lite/` | Latent diffusion, after Medfusion (Müller-Franzes et al., 2023) | 7 | `python train.py --stage vae ...` then `--stage ldm ...` | 32 | 62 |
| `panodiff_hr/` | PanoDiff's U-Net at full resolution | 8 | `python train_hr.py --out runs/panodiff_hr --epochs 110 --batch 1` | 4 | 120 |

The global batch is 32 for the StyleGANs (NVIDIA's value) and 64 for FastGAN (the batch size of the
original FastGAN run behind the two-stage arm). Training scripts launch one process per GPU through
SLURM (`SLURM_PROCID`, `SLURM_NTASKS`) and run unchanged on one GPU; `slurm_example.sh` shows the
layout used. Run them from the repository root, e.g. `python baselines/stylegan3_lite/train.py ...`.

## Notes per model

- **StyleGAN2-ADA** is re-implemented in plain PyTorch (`networks.py`, `augment.py`) because NVIDIA's
  custom CUDA kernels do not build on ROCm. Parameter counts match the reference implementation.
- **StyleGAN3-T** imports NVIDIA's own mapping, Fourier-input and synthesis layers from the official
  [stylegan3](https://github.com/NVlabs/stylegan3) repository (clone it and set `STYLEGAN3_REPO`
  to the clone; on ROCm run `patch_stylegan_rocm.py` once), and replaces only the fused
  `filtered_lrelu` kernel and the square
  canvas. `test_equivalence.py` checks the replacement against NVIDIA's reference path
  (outputs agree to 7e-7, gradients to 8e-5 relative).
- **FastGAN-HR** uses FastGAN's generator and discriminator at native resolution.
- **ADM** is OpenAI's guided-diffusion with a small patch for SLURM launching and a 1024 × 1024
  channel multiplier (`adm/guided_diffusion_slurm.patch`).

## Sampling

`sample.py` writes 7243 PNGs at 1024 × 512 for any trained arm, with seeds derived from the image
index so the output does not depend on the number of GPUs:

```bash
python sample.py --model sg3 --out samples/sg3 --batch 4
```

The FID, IS, precision/recall, ViT detectability, anatomy and memorisation scripts that score these
samples are in [`evaluation/`](../evaluation).
