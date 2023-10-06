#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=attack

dataset="coco"
attack_name="pgd"
shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/${dataset}_logs/noattack_$SLURM_JOB_ID.log"

mkdir -p "/${shared_folder}/adv_llava/results/${dataset}_logs"
exec &> ./main_output.out

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA

### Run python
    ~/anaconda3/envs/adv_env/bin/python3.9 main.py \
    --model-name "${shared_folder}/ckpts/LLAMA-on-LLaVA" \
    --dataset "coco" \
    --image_size 224 \
    --subset 50 \
    --llava_temp 0.1 \
    --temp 0.1 \
    --save_image 'False' \
    --attack_name "pgd" \
    --lr 0.01 \
    --eps 0.5 \
    --grad_sparsity 99 \
    --nb_iter 30 \
    --norm inf \
    --binary_search_steps 5 \
    --query "Fill in the blank of five templates with single sentence regarding this image. Please follow the format as - 1.Content:{}, 2.Background:{}, 3.Composition:{}, 4.Attribute:{}, 5.Context:{}" \
    --use_descriptors 'False'

echo "Ending time: $(date)" 

rm "./slurm-${SLURM_JOB_ID}.out"
