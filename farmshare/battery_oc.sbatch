#!/bin/bash
#SBATCH --job-name=marc-bat-oc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/batoc_%A_%a.log
#SBATCH --requeue
# Overcooked confirmation battery (30 runs). --array=0-29%4
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python battery_oc_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "batoc $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
python run.py --config configs/marc_overcooked.yaml \
  --network-kind "$KIND" --layout "$LAYOUT" --seed "$SEED" \
  --tag "$TAG" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE batoc $SLURM_ARRAY_TASK_ID ($TAG $LAYOUT seed$SEED) ==="
