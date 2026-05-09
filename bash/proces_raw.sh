#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GPU CSV → Parquet ETL (configs/gpuETL.yaml, src/etl/gpu.py)
#
# Run locally:
#   bash scripts/gpu_etl.sh run
#   bash scripts/gpu_etl.sh traces
#   bash scripts/gpu_etl.sh traces --lower_bound_sec 300 --upper_bound_sec 7200
#
# Submit as a CPU-heavy batch job (edit account/partition/QOS for your cluster):
#   cd /scratch/aa9360/supercloud_power
#   sbatch scripts/gpu_etl.sh run
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gpu-etl
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=h200_public
#SBATCH --qos=gpu48
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/aa9360/supercloud_power/output/gpu-etl/slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

mkdir -p output/gpu-etl

PYTHON="${PYTHON:-/share/apps/anaconda3/2025.06/bin/python3}"

if [ ! -f bash/output/reprocess-log.txt ]; then
    mkdir -p bash/output
    touch bash/output/reprocess-log.txt
fi


yn_prompt() {
    local msg="$1"
    local reply
    while true; do
        read -r -p "${msg} y=reprocess all, n=traces only, q=quit: " reply
        case "${reply,,}" in
            q | quit) 
                echo "Exiting."
                exit 0 
                ;;  # quit
            y | yes) 
                "$PYTHON" -u src/etl/gpu.py --reprocess 2>&1 | tee -a bash/output/reprocess-log.txt || true
                gpu_exit=${PIPESTATUS[0]}
                "$PYTHON" -u src/etl/slurm.py 2>&1          | tee -a bash/output/reprocess-log.txt || true
                slurm_exit=${PIPESTATUS[0]}
                "$PYTHON" -u src/etl/seq.py --reprocess 2>&1 | tee -a bash/output/reprocess-log.txt || true
                seq_exit=${PIPESTATUS[0]}
                exit $seq_exit || $gpu_exit || $slurm_exit
                ;;  
            n | no | "") 
                "$PYTHON" -u src/etl/gpu.py --build_traces 2>&1 | tee -a bash/output/reprocess-log.txt || true
                gpu_exit=${PIPESTATUS[0]}
                "$PYTHON" -u src/etl/slurm.py 2>&1               | tee -a bash/output/reprocess-log.txt || true
                slurm_exit=${PIPESTATUS[0]}
                "$PYTHON" -u src/etl/seq.py --reprocess 2>&1   | tee -a bash/output/reprocess-log.txt || true
                seq_exit=${PIPESTATUS[0]}
                exit $seq_exit || $gpu_exit || $slurm_exit
                ;;  
            *) echo "Invalid input. Use y, n, or q." ;;  # invalid input
        esac
    done
}

yn_prompt "Select mode: "