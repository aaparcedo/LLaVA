#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=1:00:00
#SBATCH --job-name=llava_gen


shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/llavav2_response_$SLURM_JOB_ID.log"
blip2="Salesforce/blip2-opt-2.7b"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
llavav2="liuhaotian/llava-v1.5-13b"
clip="openai/clip-vit-large-patch14"

orig_image_folder="/groups/sernam/datasets/coco/val2014"
adv_image_folder="/groups/sernam/adv_llava/adv_datasets/coco/269126_clip_pgd_eps0.2_nbiter30"
adv_clip336_image_folder="/groups/sernam/adv_llava/adv_datasets/coco/269389_clip336_pgd_eps0.2_nbiter50"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 generate_answers_llavav2.py \
    --model-path "$llavav2" \
    --image-folder "$adv_clip336_image_folder" \
    --question-file "/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_llava.jsonl" \
    --subset 1000 \
    --temperature 0

echo "Ending time: $(date)" 

