#!/bin/bash
#SBATCH --job-name=marc-refactor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/users/flukol/marc/logs/marc_refactor_%j.log
#SBATCH --requeue
# Refactor-correctness check: vanilla network_kind on cramped_room must
# reproduce the validated baseline (~240 sparse). Short limit -> backfills
# fast (validation check, not a training run).
set -e
SCRATCH=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $SCRATCH/envs/jaxmarl
cd $SCRATCH/code/marc
nvidia-smi --query-gpu=name --format=csv,noheader
python run.py --config configs/marc_overcooked.yaml \
  --layout cramped_room --network-kind vanilla \
  --results-dir $SCRATCH/results --gifs-dir $SCRATCH/gifs --tag refactor
echo "=== REFACTOR CHECK DONE ==="
