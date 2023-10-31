#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=5:00:00
#SBATCH --job-name=tvqa_llava

shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/textvqa/orig_llava1.5_response_$SLURM_JOB_ID.log"


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
~/anaconda3/envs/adv_env/bin/python3.9 -m textvqa.generate_answers_llava2 \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /groups/sernam/datasets/vqa/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /groups/sernam/datasets/vqa/textvqa/train_images \
    --answers-file /groups/sernam/adv_llava/results/responses/textvqa/orig_llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/textvqa/eval_textvqa.py \
    --annotation-file /groups/sernam/datasets/vqa/textvqa/TextVQA_0.5.1_val.json \
    --result-file /groups/sernam/adv_llava/results/responses/textvqa/orig_llava-v1.5-13b.jsonl

echo "Ending time: $(date)" 




