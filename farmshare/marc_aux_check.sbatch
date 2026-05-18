#!/bin/bash
#SBATCH --job-name=marc-aux-chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:50:00
#SBATCH --output=/scratch/users/flukol/marc/logs/marc_aux_%j.log
#SBATCH --requeue
# Task 7 fast validation @ reduced budget (1.5e6): confirm aux losses + the
# 3 ablation flags all run stably with the compute_aux optimization, and
# measure optimized per-run cost to size the Task 9 sweep. Not a science run.
set -e
S=/scratch/users/flukol/marc
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate $S/envs/jaxmarl
cd $S/code/marc
nvidia-smi --query-gpu=name --format=csv,noheader
C="--config configs/marc_overcooked.yaml --layout cramped_room \
   --total-timesteps 1000000 --results-dir $S/results --gifs-dir $S/gifs"
echo "=== marc full ==="
python run.py $C --network-kind marc --tag aux2_full
echo "=== ablation lambda_aux=0 ==="
python run.py $C --network-kind marc --tag aux2_lam0 --set LAMBDA_AUX=0
echo "=== ablation beta=0 ==="
python run.py $C --network-kind marc --tag aux2_beta0 --set BETA=0
echo "=== ablation zero teammate ==="
python run.py $C --network-kind marc --tag aux2_zt --set ZERO_TEAMMATE=true
echo "=== MARC-AUX CHECK DONE ==="
