#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=3:00:00
#SBATCH --job-name=txtvqa_llava

shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/textvqa/caption_llava1.5_$SLURM_JOB_ID.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 gen_llava_caption.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-folder /groups/sernam/datasets/vqa/textvqa/train_images \
    --temperature 0 \
    --conv-mode vicuna_v1_1 \
    --prompt "What is this image about? Answer in one sentence."

echo "Ending time: $(date)" 




