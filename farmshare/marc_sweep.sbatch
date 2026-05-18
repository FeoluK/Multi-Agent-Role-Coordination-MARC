#!/bin/bash
#SBATCH --job-name=marc-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/sweep_%A_%a.log
#SBATCH --requeue
# Task 9 sweep. One array task = one (config, layout, seed) run @ full 5e6
# + eval. 48h ceiling per task (real science runs; actual ~1-15 min). QOS
# caps GPU jobs at 4 concurrent -> submit with %4.
#   Core comparison:  --array=0-29%4   (vanilla + marc, 5 layouts, 3 seeds)
#   Ablations later:  --array=30-74%4
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python sweep_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
nvidia-smi --query-gpu=name --format=csv,noheader
python run.py --config configs/marc_overcooked.yaml \
  --network-kind "$KIND" --layout "$LAYOUT" --seed "$SEED" \
  --tag "sweep_${TAG}" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE task $SLURM_ARRAY_TASK_ID ($TAG $LAYOUT seed$SEED) ==="
