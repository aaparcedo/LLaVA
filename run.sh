#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=unt-llava


shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/classification_$SLURM_JOB_ID.log"
blip2="Salesforce/blip2-opt-2.7b"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
clip="openai/clip-vit-large-patch14"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 main.py \
    --model-name "['$clip', '$llava']" \
    --dataset "['imagenet']" \
    --image_size 224 \
    --subset 1000 \
    --llava_temp 0.1 \
    --temp 0.1 \
    --save_image 'False' \
    --attack_name "pgd" \
    --lr 0.01 \
    --eps "[0.4]" \
    --grad_sparsity 99 \
    --nb_iter 30 \
    --adv_path None \
    --norm inf \
    --targeted True \
    --binary_search_steps 5 \
    --query '[descriptor_prompt]'\
    --use_descriptors 'True' \
    --attack_descriptors 'True'


echo "Ending time: $(date)" 


