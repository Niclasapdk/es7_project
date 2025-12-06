#!/bin/bash
#SBATCH --job-name=gnss_tcn
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=70
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=logs.out
#SBATCH --error=logs.err

set -euo pipefail

export PYTHONUNBUFFERED=1
# Prevent CPU oversubscription when DDP spawns 8 procs × 16 workers each
export OMP_NUM_THREADS=1

# 128 CPUs ≈ 16 workers/GPU → --workers 16 (per process). If that overloads the FS, drop to 12.
singularity exec --nv /ceph/container/pytorch/pytorch_25.09.sif \
  torchrun --standalone --nproc_per_node=4 chat_tcn_train.py \
  --data gnss_sweptcw_500k.npz \
  --ckpt-dir ckpts_curriculum \
  --epochs 160 \
  --batch 64 \
  --workers 16 \
  --width 128 \
  --blocks 16 \
  --kernel 9 \
  --dropout 0.05 \
  --lr 6e-5 \
  --wd 1e-4 \
  --grad-clip 1.0 \
  --ema 0.9997 \
  --amp \
  --scheduler cawr \
  --cawr-T0 5 \
  --cawr-Tmult 2 \
  --warmup-frac 0.05 \
  --time-w 1.0 \
  --spec-w 0.3 \
  --spec-w-final 0.6 \
  --spec-w-ramp-epochs 10 \
  --spec-w-in 1.0 \
  --spec-w-guard 1.0 \
  --spec-w-out 1.0 \
  --smooth-w 0.05 \
  --notch-peaks 1 \
  --notch-q 600 \
  --notch-depth 60 \
  --perseq-norm \
  --align-maxlag 0 \
  --align-w 0.0 \
  --align-frac-steps 0 \
  --diag-ai-evm epoch \
  --curriculum \
  --curriculum-epochs 40 \
  --curriculum-min-frac 0.3