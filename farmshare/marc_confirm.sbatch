#!/bin/bash
#SBATCH --job-name=marc-confirm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/conf_%A_%a.log
#SBATCH --requeue
# No-regression confirm: abl_lam0 (LAMBDA_AUX=0) on cramped_room+coord_ring,
# 3 seeds. --array=0-5%4
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python confirm_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "conf $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
python run.py --config configs/marc_overcooked.yaml \
  --network-kind "$KIND" --layout "$LAYOUT" --seed "$SEED" \
  --tag "$TAG" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE conf $SLURM_ARRAY_TASK_ID ($TAG $LAYOUT seed$SEED) ==="
