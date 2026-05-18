#!/bin/bash
#SBATCH --job-name=marc-ippo-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/marc/logs/ippo_baseline_%j.log
#SBATCH --requeue

# Vanilla IPPO baseline on the 5 standard Overcooked layouts.
# cramped_room first: it is the primary go/no-go (~200 sparse return expected).
# Each layout ~5-15 min on GPU; all 5 well under the 48h cap.

set -e
SCRATCH=/scratch/users/flukol
PROJ=$SCRATCH/marc
REF=$PROJ/code/JaxMARL_ref

source $SCRATCH/miniconda_fresh/etc/profile.d/conda.sh
conda activate $PROJ/envs/jaxmarl
cd $PROJ

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

for LAYOUT in cramped_room asymm_advantages coord_ring forced_coord counter_circuit; do
  echo "=================== $LAYOUT ==================="
  python code/baselines/run_ippo_overcooked.py \
    --layout "$LAYOUT" \
    --jaxmarl-ref "$REF" \
    --results-dir "$PROJ/results" \
    --gifs-dir "$PROJ/gifs"
done
echo "=================== ALL LAYOUTS DONE ==================="
