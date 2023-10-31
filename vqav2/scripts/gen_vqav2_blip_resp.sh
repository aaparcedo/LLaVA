#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --job-name=blip2_gen
#SBATCH --constraint=h100



shared_folder="/groups/sernam"
run_name="instruct13b_adv274351"
export log_file="/${shared_folder}/adv_llava/results/logs/vqav2/${run_name}_response_$SLURM_JOB_ID.log"
blip2="blip2_opt"
instructblip="blip2_vicuna_instruct"

orig_image_folder="/groups/sernam/datasets/coco/val2014"
adv_image_folder="/groups/sernam/adv_llava/adv_datasets/coco/classification/274351_blip2_eps0.2_nbiter50"

exec &> $log_file

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

nvidia-smi

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/vqav2/generate_vqa_answer_blip.py \
    --model-path "$instructblip" \
    --model-type "vicuna13b" \
    --image-folder "$adv_image_folder" \
    --image_ext pt \
    --data-file "/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_subset1000.jsonl" \
    --prompt_format "Question: {} Short answer:"

echo "Ending time: $(date)" 

