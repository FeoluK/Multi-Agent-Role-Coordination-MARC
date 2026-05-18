#!/bin/bash
#SBATCH --job-name=marc-ippo-gng
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/users/flukol/marc/logs/ippo_gonogo_%j.log
#SBATCH --requeue
# Fast go/no-go: cramped_room only. Short limit -> backfills fast in busy queue.
# Expected sparse return ~200; primary check for reward-scale (+20 vs +1) bugs.
set -e
SCRATCH=/scratch/users/flukol; PROJ=$SCRATCH/marc
source $SCRATCH/miniconda_fresh/etc/profile.d/conda.sh
conda activate $PROJ/envs/jaxmarl
cd $PROJ
nvidia-smi --query-gpu=name,memory.total --format=csv
python code/baselines/run_ippo_overcooked.py --layout cramped_room \
  --jaxmarl-ref $PROJ/code/JaxMARL_ref \
  --results-dir $PROJ/results --gifs-dir $PROJ/gifs
echo "=== GO/NO-GO DONE ==="
