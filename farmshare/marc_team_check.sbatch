#!/bin/bash
#SBATCH --job-name=marc-team-chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/users/flukol/marc/logs/marc_team_%j.log
#SBATCH --requeue
# Regression (vanilla ~240, marc_self ~vanilla) + NEW: kind=marc must run
# and learn on cramped_room. Short limit -> backfills fast (validation).
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
nvidia-smi --query-gpu=name --format=csv,noheader
for K in vanilla marc_self marc; do
  echo "=== $K ==="
  python run.py --config configs/marc_overcooked.yaml --layout cramped_room \
    --network-kind $K --results-dir $S/results --gifs-dir $S/gifs --tag teamchk
done
echo "=== MARC-TEAM CHECK DONE ==="
