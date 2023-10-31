#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=h100
#SBATCH --time=2:00:00
#SBATCH --job-name=sqa_blip2

shared_folder="/groups/sernam"
run_name="instructblip"
export log_file="/${shared_folder}/adv_llava/results/logs/sqa/${run_name}_response_$SLURM_JOB_ID.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 
### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID

rm "./slurm-${SLURM_JOB_ID}.out"

~/anaconda3/envs/adv_env/bin/python3.9 -m sqa.generate_answers_sqa_blip \
    --model-path blip2_vicuna_instruct \
    --model-type vicuna13b \
    --question-file /groups/sernam/datasets/vqa/scienceqa/llava_test_CQM-A.json \
    --image-folder /groups/sernam/datasets/vqa/scienceqa/test \
    --answers-file /groups/sernam/adv_llava/results/responses/sqa/${run_name}/${run_name}.jsonl \
    --image_ext jpg \
    --single-pred-prompt 

~/anaconda3/envs/adv_env/bin/python3.9 ~/projects/LLaVA/sqa/evaluate_sqa_llava2.py \
    --base-dir /groups/sernam/datasets/vqa/scienceqa \
    --result-file /groups/sernam/adv_llava/results/responses/sqa/${run_name}/${run_name}.jsonl \
    --output-file /groups/sernam/adv_llava/results/responses/sqa/${run_name}/${run_name}_output.jsonl \
    --output-result /groups/sernam/adv_llava/results/responses/sqa/${run_name}/${run_name}_result.json

echo "Ending time: $(date)" 




