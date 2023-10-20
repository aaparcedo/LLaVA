#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=15:00:00
#SBATCH --job-name=gen_adv
#SBATCH --constraint=h100


shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/gen_adv_$SLURM_JOB_ID.log"
blip2="Salesforce/blip2-opt-2.7b"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
clip="openai/clip-vit-large-patch14"
clip336="openai/clip-vit-large-patch14-336"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 generate_adv_datasets.py \
    --model-name "['$clip336']" \
    --dataset "['imagenet']" \
    --image-folder "/groups/sernam/datasets/imagenet/val" \
    --annotation-file "/groups/sernam/datasets/imagenet_val2012.jsonl" \
    --image_size 336 \
    --save_image 'False' \
    --attack_name "pgd" \
    --subset 10000000 \
    --lr 0.01 \
    --eps "[0.2]" \
    --grad_sparsity 99 \
    --nb_iter 50 \
    --norm inf \
    --targeted False \
    --binary_search_steps 5 \
    --use_descriptors 'False' \
    --batch_size 16

echo "Ending time: $(date)" 


