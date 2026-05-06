#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Supercloud SBATCH training script for gpu-power-model (DiT-AR v5)
#
# Submit with:
#   cd /scratch/aa9360/supercloud_power
#   sbatch scripts/train.sh
#
# The job runs the Python training process directly — NOT "sleep infinity +
# interactive".  When training exits (success, crash, or SIGTERM), the SLURM
# job ends immediately, so GPUs are never left idle.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gpu-power-model
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=h200_public
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/aa9360/supercloud_power/output/v5/slurm-%j.out
#SBATCH --open-mode=append

# Send SIGUSR1 to the Python process 120 s before the wall-time limit so
# training can save a checkpoint and exit cleanly (our signal handler catches
# it).  Without this, SLURM sends SIGTERM with zero notice.
#SBATCH --signal=SIGUSR1@120

set -uo pipefail

PROJ=/scratch/aa9360/supercloud_power
cd "$PROJ"

# Create output directory before the job starts (SLURM needs it for --output).
mkdir -p output/v5

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "=== CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NOT SET} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== $(python -c 'import torch; print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")') ==="

# Kill any leftover Python processes from a previous run that might have
# leaked CUDA contexts.  This prevents the pre-flight OOM check in _ddp_setup
# from failing with "GPU N has only X GB free".
echo "=== Clearing any zombie python processes from previous runs ==="
pkill -9 -u "$USER" -f 'python.*main' 2>/dev/null || true
sleep 3   # give CUDA contexts time to be released

echo "=== Rebuilding chunk index (stride/window change — skips NPY re-processing) ==="
/share/apps/anaconda3/2025.06/bin/python3 -u src/etl/seq.py --chunks-only 2>&1 | tee -a output/v5/output.txt

echo "=== Starting training ==="
/share/apps/anaconda3/2025.06/bin/python3 -u src/model/main.py 2>&1 | tee -a output/v5/output.txt
TRAIN_EXIT=$?

echo "=== Training exited with code $TRAIN_EXIT at $(date) ==="
exit $TRAIN_EXIT
