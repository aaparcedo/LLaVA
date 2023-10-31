#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=3:00:00
#SBATCH --job-name=sqa_llava

shared_folder="/groups/sernam"
export log_file="/${shared_folder}/adv_llava/results/logs/sqa/adv272351_llava1.5_response_$SLURM_JOB_ID.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

Run python
~/anaconda3/envs/adv_env/bin/python3.9 -m sqa.generate_answers_sqa_llava \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /groups/sernam/datasets/vqa/scienceqa/llava_test_CQM-A.json \
    --image-folder /groups/sernam/adv_llava/adv_datasets/sqa/retrieval/275041_clip336_eps0.2_nbiter50 \
    --answers-file /groups/sernam/adv_llava/results/responses/sqa/llava1.5/adv275041_llava-v1.5-13b.jsonl \
    --image_ext pt \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1_1

~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/sqa/evaluate_sqa_llava2.py \
    --base-dir /groups/sernam/datasets/vqa/scienceqa \
    --result-file /groups/sernam/adv_llava/results/responses/sqa/llava1.5/adv275041_llava-v1.5-13b.jsonl \
    --output-file /groups/sernam/adv_llava/results/responses/sqa/llava1.5/adv275041_llava-v1.5-13b_output.jsonl \
    --output-result /groups/sernam/adv_llava/results/responses/sqa/llava1.5/adv275041_llava-v1.5-13b_result.json

echo "Ending time: $(date)" 




