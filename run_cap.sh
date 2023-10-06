#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --job-name=attack
#SBATCH --constraint=h100


dataset="coco"
attack_name="pgd"
shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/retrieval/coco_clip_recall@k_$SLURM_JOB_ID.log"

mkdir -p "/${shared_folder}/adv_llava/results/${dataset}_logs"
exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA

### Run python
    ~/anaconda3/envs/adv_env/bin/python3.9 eval_cap_recall.py \
	--task caption \
    --dataset $dataset \
    --attack_name $attack_name \
    --first_response_only False \
    --save_image False

rm "./slurm-${SLURM_JOB_ID}.out"
