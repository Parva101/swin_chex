#!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-16:00:00
#SBATCH -p general
#SBATCH -q public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="psheta12@asu.edu"
#SBATCH --export=NONE

module load mamba
source activate venv   # or: conda activate venv
cd ~/SwinCheX
IMG_PATH=/scratch/psheta12/CXR8/images/images
VAL_FRACTION=0.1
SPLIT_SEED=42

python train_eval.py --mode train --images_path "$IMG_PATH" --val_fraction $VAL_FRACTION --split_seed $SPLIT_SEED
python train_eval.py --mode eval  --images_path "$IMG_PATH"