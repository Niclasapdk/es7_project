#!/bin/bash
#SBATCH --job-name=gnss_tcn_8g
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=logs.out
#SBATCH --error=logs.err

set -euo pipefail

export PYTHONUNBUFFERED=1
# Prevent CPU oversubscription when DDP spawns 8 procs × 16 workers each
export OMP_NUM_THREADS=1

# 128 CPUs ≈ 16 workers/GPU → --workers 16 (per process). If that overloads the FS, drop to 12.
singularity exec --nv /ceph/container/pytorch/pytorch_25.09.sif torchrun --standalone --nproc_per_node=8 train_tcn.py --data gnss_l1_sweptcw.npz --epochs 100 --W 1024 --H 256 --batch 64 --width 48 --blocks 8 --kernel 5 --workers 16 --input-mode dualpath --prefilter stft_gate --phase-pen 0.08 --gain-pen 0.02 --headline raw --select-metric snr_out --amp --lr 3e-4 --weight-decay 1e-4 --ema --ema-decay 0.999 --eval-use-ema --save-ema --out ckpts/tcn_dual_raw_8g.pt --log-csv logs/train_diag.csv --progress --log-every 500 --eval-first --align-eval --headline aligned --print-align-stats --select-metric snr_out_star
