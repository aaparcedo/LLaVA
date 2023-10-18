#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=4:00:00
#SBATCH --job-name=llava_gen


shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/llava_response_$SLURM_JOB_ID.log"
blip2="Salesforce/blip2-opt-2.7b"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
clip="openai/clip-vit-large-patch14"

orig_image_folder="/groups/sernam/datasets/coco/val2014"
adv_image_folder="/groups/sernam/adv_llava/adv_datasets/coco/269126_clip_pgd_eps0.2_nbiter50"


exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 generate_answers_vqav2.py \
    --subset 1000 \
    --image_folder "$adv_image_folder" \
    --output_file "/groups/sernam/adv_llava/results/responses/vqav2/llava_response_$SLURM_JOB_ID.jsonl" 

echo "Ending time: $(date)" 


