#!/bin/bash
#SBATCH --job-name=marc-mpe-scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/mpescale_%A_%a.log
#SBATCH --requeue
# MPE scaling: vanilla vs MARC-arch (aux=0) x N=3/6/9 x 3 seeds. --array=0-17%4
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python mpe_scaling_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "mpescale $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
python run.py --config configs/marc_mpe.yaml \
  --network-kind "$KIND" --seed "$SEED" \
  --tag "${TAG}_${ADAPTER}" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE mpescale $SLURM_ARRAY_TASK_ID ($TAG $ADAPTER seed$SEED) ==="
