#!/bin/bash
#SBATCH --job-name=marc-self-chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/users/flukol/marc/logs/marc_self_%j.log
#SBATCH --requeue
# (1) vanilla regression after train_marc.py rewrite -> must still ~240.
# (2) marc_self sanity: history path runs + learns competently on cramped_room.
# Short limit -> backfills fast (validation, not the real sweep).
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== (1) vanilla regression ==="
python run.py --config configs/marc_overcooked.yaml --layout cramped_room \
  --network-kind vanilla --results-dir $S/results --gifs-dir $S/gifs --tag selfchk
echo "=== (2) marc_self sanity ==="
python run.py --config configs/marc_overcooked.yaml --layout cramped_room \
  --network-kind marc_self --results-dir $S/results --gifs-dir $S/gifs --tag selfchk
echo "=== MARC-SELF CHECK DONE ==="
