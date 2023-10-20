#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=eval-llava


shared_folder="/groups/sernam"
export run_name="cls_llava_imgnet2012val1000"
export log_file="/${shared_folder}/adv_llava/results/${run_name}_${SLURM_JOB_ID}.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA
export SLURM_JOB_ID=$SLURM_JOB_ID



rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
~/anaconda3/envs/adv_env/bin/python3.9 evaluate_classification.py \
    --model-name "['llava']" \
    --dataset "['imagenet']" \
    --image-folder "/groups/sernam/datasets/imagenet/val" \
    --annotation-file "/groups/sernam/datasets/imagenet_val2012_subset1000.jsonl" \
    --subset  \
    --llava_temp 0.1 \
    --temp 0.1 \
    --task "classification" \
    --batch_size 16 \
    --query '[one_word_response]'\

echo "Ending time: $(date)" 


