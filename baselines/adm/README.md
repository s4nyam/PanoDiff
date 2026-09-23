# ADM baseline

ADM is OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) at commit `22e0df8`,
with `guided_diffusion_slurm.patch` applied. The patch replaces the MPI launcher with a
SLURM / `torch.distributed` one (same API), shards the data by rank, and adds a channel multiplier
for 1024 × 1024 images.

```bash
git clone https://github.com/openai/guided-diffusion.git && cd guided-diffusion
git checkout 22e0df8 && git apply ../guided_diffusion_slurm.patch
PYTHONPATH=$PWD python scripts/image_train.py --data_dir <1024x1024 corpus> \
  --image_size 1024 --num_channels 128 --num_res_blocks 2 --attention_resolutions 32,16,8 \
  --learn_sigma True --class_cond False --diffusion_steps 1000 --noise_schedule cosine \
  --use_scale_shift_norm True --resblock_updown True --use_fp16 True \
  --lr 1e-4 --batch_size 1 --lr_anneal_steps 24898 --save_interval 2000
```

ADM's code is square-only, so it was trained on the 1024 × 512 corpus resized to 1024 × 1024, and its
samples are resized back to 1024 × 512 by `../sample.py`. 24,898 steps at global batch 32 is the
110-epoch budget.
