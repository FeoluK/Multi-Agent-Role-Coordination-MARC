#!/bin/bash
#SBATCH --job-name=marc-bat-mpe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/batmpe_%A_%a.log
#SBATCH --requeue
# Loss-fix battery on MPE (28 runs). --array=0-27%4
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python battery_mpe_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "batmpe $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
python run.py --config configs/marc_mpe.yaml \
  --network-kind "$KIND" --seed "$SEED" \
  --tag "${TAG}_${ADAPTER}" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE batmpe $SLURM_ARRAY_TASK_ID ($TAG $ADAPTER seed$SEED) ==="
