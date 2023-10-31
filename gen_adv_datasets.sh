#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
#SBATCH --job-name=instblip
#SBATCH --constraint=h100

shared_folder="/groups/sernam"
run_name="eval_adv274351_instructblip"
dataset="textvqa"
task="classification"

mkdir -p /${shared_folder}/adv_llava/results/logs/$dataset/$task

export log_file="/${shared_folder}/adv_llava/results/logs/$dataset/$task/${run_name}_${SLURM_JOB_ID}.log"
blip2="Salesforce/blip2-opt-2.7b"
blip2itm="blip2_image_text_matching"
llava="/groups/sernam/ckpts/LLAMA-on-LLaVA"
llava2="liuhaotian/llava-v1.5-13b"
clip="openai/clip-vit-large-patch14"
clip336="openai/clip-vit-large-patch14-336"
orig_coco="/groups/sernam/datasets/coco/val2014"
instructblip="blip2_vicuna_instruct"
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
    --model-path "['$instructblip']" \
    --dataset "['$dataset']" \
    --model-type vicuna13b \
    --image-folder "/groups/sernam/adv_llava/adv_datasets/coco/classification/274351_blip2_eps0.2_nbiter50" \
    --data-file "/groups/sernam/datasets/vqa/vqav2/coco2014val_questions_subset1000.jsonl" \
    --save_image 'False' \
    --image_ext 'pt' \
    --task $task \
    --use_ce_loss 'True' \
    --attack_name "None" \
    --lr 0.01 \
    --eps "[0.2]" \
    --grad_sparsity 99 \
    --nb_iter 50 \
    --norm inf \
    --targeted False \
    --binary_search_steps 5 \
    --use_descriptors 'False' \
    --batch_size 8 \
    --num_workers 2 \

echo "Ending time: $(date)" 


