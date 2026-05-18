#!/bin/bash
#SBATCH --job-name=marc-cds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/cds_%A_%a.log
#SBATCH --requeue
# CDS-style diversity baseline sweep (identity-MI intrinsic reward). MPE N=3/6/9 + OC 5 layouts,
# 3 seeds. Lean (32G/4cpu/1gpu) for backfill.
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python cds_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "cds $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
LAYOUT_ARG=""
[ -n "$LAYOUT" ] && LAYOUT_ARG="--layout $LAYOUT"
python run.py --config "configs/$CONFIG" \
  --network-kind "$KIND" --seed "$SEED" $LAYOUT_ARG \
  --tag "$TAG" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE cds $SLURM_ARRAY_TASK_ID ($TAG $KIND seed$SEED) ==="
