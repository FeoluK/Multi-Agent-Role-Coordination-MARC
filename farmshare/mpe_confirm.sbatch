#!/bin/bash
#SBATCH --job-name=marc-mpe-cf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/mpecf_%A_%a.log
#SBATCH --requeue
# normgateup confirmation: vanilla/arch/normgateup x N=3/6/9 x 3 seeds.
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
SPEC=$(python mpe_confirm_manifest.py "$SLURM_ARRAY_TASK_ID")
echo "mpecf $SLURM_ARRAY_TASK_ID -> $SPEC"
eval "$SPEC"
python run.py --config configs/marc_mpe.yaml \
  --network-kind "$KIND" --seed "$SEED" \
  --tag "${TAG}_${ADAPTER}" $SETS \
  --results-dir "$S/results" --gifs-dir "$S/gifs"
echo "=== DONE mpecf $SLURM_ARRAY_TASK_ID ($TAG $ADAPTER seed$SEED) ==="
