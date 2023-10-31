#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=llava1.5cap
#SBATCH --constraint=h100

run_name="adv274350_llava1.5_13b"
shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/coco/retrieval/${run_name}_$SLURM_JOB_ID.log"

blip2="blip2-opt"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
llava2="liuhaotian/llava-v1.5-13b"
clip336="openai/clip-vit-large-patch14-336"
clip="openai/clip-vit-large-patch14"
instructblip="blip2_vicuna_instruct"
blip2itm="blip2_image_text_matching"

orig_folder="/groups/sernam/datasets/coco/val2014"
adv_folder="/groups/sernam/adv_llava/adv_datasets/coco/classification/274350_clip336_eps0.2_nbiter50"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA

### Run python
    ~/anaconda3/envs/adv_env/bin/python3.9 evaluate_caption_retrieval.py \
    --model-path $llava2 \
    --model-type vicuna13b \
    --data-file /groups/sernam/datasets/coco/coco_2014val_caption_subset1000.json \
    --image_ext pt \
    --temperature 0 \
    --image-folder $adv_folder \
    --prompt "generate a short caption of this image" \
    --num_beams 1 \

rm "./slurm-${SLURM_JOB_ID}.out"
