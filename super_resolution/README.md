# Stage 2 — HAT-SR (4× super-resolution)

The released Real-HAT-GAN ×4 model (Chen et al.) is fine-tuned, without architectural change, on the
7243 high-resolution radiographs (paper Sections 3.3–3.4). The loss is L1 + VGG19 perceptual +
adversarial (weight 0.1) with a U-Net discriminator with spectral normalisation; the low-resolution
inputs are produced on the fly by the two-stage Real-ESRGAN degradation pipeline (blur, noise, JPEG).

| Setting | Value |
|---|---|
| Training input | random 256 × 256 crops of the HR radiographs (64 × 64 degraded inputs) |
| Batch size | 4 |
| Schedule | 400,000 iterations (≈ 221 epochs) |
| Optimiser | Adam, lr 1e-4 for G and D, EMA 0.999 |
| Inference | the whole 256 × 128 PanoDiff output → 1024 × 512 in one pass, no padding or rescaling |

## Train

```bash
# 1. put the HR corpus in datasets/images/HR and write the file list the loader reads
mkdir -p datasets/images && ln -s ../../../work/corpus/HR datasets/images/HR
python tools/create_meta.py && mkdir -p hat/data/meta_info && mv meta_info.txt hat/data/meta_info/
# 2. the released weights go to experiments/pretrained_models/Real_HAT_GAN_SRx4.pth
python train.py -opt options/train_Real_HAT_GAN_SRx4_finetune_PR.yml
```

## Upscale PanoDiff samples

```bash
python test.py -opt options/test_HAT_GAN_Real_SRx4_PR.yml     # dataroot_lq: folder of 256 x 128 samples
```

`tools/merge.py` and `tools/unmerge.py` gather the per-seed sample folders into one folder and back;
`tools/create_lr.py` writes 2× and 4× area-downsampled copies of an HR folder.

## SwinIR comparison (Section 4.5.1)

`swinir/infer_swinir_ft.py` runs the SwinIR that was fine-tuned on fixed fourfold area-downsampled
pairs of the same radiographs. The restoration test on 185 held-out DENTEX radiographs is in
[`evaluation/sr_restoration`](../evaluation/sr_restoration).
