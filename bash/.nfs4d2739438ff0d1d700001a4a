#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Supercloud SBATCH inference script for gpu-power-model (DiT-AR v5)
#
# Submit with:
#   cd /scratch/aa9360/supercloud_power
#   sbatch scripts/inference.sh
#
# Generates synthetic power traces for the test set, writes per-job .npy
# files to data/inference/v5/synth/, a summary CSV, and a samples.png plot.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gpu-power-infer
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=h200_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/aa9360/supercloud_power/output/v5/infer-%j.out
#SBATCH --open-mode=append

set -uo pipefail

PROJ=/scratch/aa9360/supercloud_power
cd "$PROJ"

mkdir -p output/v5

echo "=== Inference job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT SET} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== $(/share/apps/anaconda3/2025.06/bin/python3 -c 'import torch; print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")') ==="

echo "=== Starting inference ==="
/share/apps/anaconda3/2025.06/bin/python3 -u src/model/inference.py \
    --config configs/v5.yaml --ckpt_dir /scratch/aa9360/supercloud_power/output/v5/ckpt \
    2>&1 | tee -a output/v5/infer-output.txt
INFER_EXIT=$?

echo "=== Inference exited with code $INFER_EXIT at $(date) ==="
exit $INFER_EXIT
