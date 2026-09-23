# Stage 1 — PanoDiff (diffusion)

A U-Net denoiser (34.0 M parameters) with self-attention at the two coarsest scales, a cosine noise
schedule over 1000 training timesteps, EMA weights and DDIM sampling (paper Section 3.2).
It generates 256 × 128 radiographs (tensors of shape 3 × 128 × 256).

| Setting | Value |
|---|---|
| Training input | 256 × 128 (images are resized internally) |
| Batch size | 4 |
| Schedule | 110 epochs (199,210 steps) |
| Optimiser | AdamW, lr 1e-4, cosine LR schedule with 300 warm-up steps |
| Sampling | DDIM, 250 steps, η = 1 |

## Train

Put the corpus images in a folder and pass it as a Hugging Face `imagefolder` dataset:

```bash
python train.py --dataset_name ../work/corpus/LR --resolution 128 --train_batch_size 4 --num_epochs 110
```

## Sample

```bash
python generate.py --pretrained_model_path <checkpoint.pth> --samples_dir ../work/samples_lr --eval_batch_size 32
python create_generated_data.py      # converts the per-seed JPEG folders to PNG and drops the grid images
```

`simple_diffusion/model/unet.py` holds the network, `simple_diffusion/scheduler/ddim.py` the sampler
and `simple_diffusion/ema.py` the weight averaging.
