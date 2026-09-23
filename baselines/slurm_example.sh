#!/bin/bash
# Example launch of a baseline on 4 nodes x 8 GPUs = 32 ranks (global batch 32 at --batch 1).
# The training scripts read their rank from SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID.
#SBATCH --job-name=sg3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00
#SBATCH --account=<your_account>
#SBATCH --partition=<gpu_partition>
set -euo pipefail
export PANODIFF_WORK=/path/to/work            # see docs/REPRODUCE.md
export STYLEGAN3_REPO=/path/to/stylegan3      # only for the StyleGAN3 baseline
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))
srun python baselines/stylegan3_lite/train.py --out "$PANODIFF_WORK/runs/sg3" --epochs 110 --batch 1 --r1-gamma 8.2
