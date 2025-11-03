#!/bin/bash
#SBATCH --job-name=gnss_tcn_8g
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs.out
#SBATCH --error=logs.err

set -euo pipefail

export PYTHONUNBUFFERED=1
# Prevent CPU oversubscription when DDP spawns 8 procs × 16 workers each
export OMP_NUM_THREADS=1

# 128 CPUs ≈ 16 workers/GPU → --workers 16 (per process). If that overloads the FS, drop to 12.
singularity exec --nv /ceph/container/pytorch/pytorch_25.09.sif torchrun --standalone --nproc_per_node=8 chat_tcn_train.py --data gnss_sweptcw_500k.npz --ckpt-dir ckpts_stage3_avg --epochs 200 --batch 64 --workers 16 --width 128 --blocks 16 --kernel 9 --dropout 0.05 --lr 6e-5 --wd 1e-4 --grad-clip 1.0 --ema 0.9997 --amp --scheduler cawr --cawr-T0 5 --cawr-Tmult 2 --warmup-frac 0.04 --notch-peaks 10 --notch-q 1200 --time-w 0.7 --smooth-w 0.008 --spec-w 0.25 --spec-w-final 1.2 --spec-w-ramp-epochs 18 --spec-w-in 3.0 --spec-w-guard 3.0 --spec-w-out 0.3 --perseq-norm --align-maxlag 32 --align-frac-steps 8 --align-w 0.7 --gain-align --peak-w 0.15 --peak-k 8 --peak-region guard --diag-ai-evm epoch
