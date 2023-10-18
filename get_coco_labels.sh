#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name=get_labels

source activate ~/anaconda3/envs/adv_env

~/anaconda3/envs/adv_env/bin/python3.9 check_coco.py


