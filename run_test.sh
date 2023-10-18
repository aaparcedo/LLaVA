#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --job-name=clip-attack

shared_folder="/groups/sernam"

mkdir -p "/${shared_folder}/adv_llava/results/attacks"
export log_file="/${shared_folder}/adv_llava/results/attacks/$(date)_$SLURM_JOB_ID.log"
export TOKENIZERS_PARALLELISM=true
exec &> test_clip_cw_out.log

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate ~/anaconda3/envs/adv_env
export PYTHONPATH=$PYTHONPATH:~/projects/LLaVA

for model_name in "openai/clip-vit-large-patch14"
do
    for dataset in "imagenet"
        do
        for confidence in 0 0.5 1
        do
            for nb_iter in 30 60
            do 
                for initial_const in 0.5 1 2 4
                do
                    echo "==> Running dataset: $dataset, attack: $attack_name, model: $model_name, use_last_n_hidden: $use_last_n_hidden"
                    ### Run python
                    ~/anaconda3/envs/adv_env/bin/python3.9 test_attacks.py \
                    --model-name $model_name \
                    --dataset $dataset \
                    --subset 1000 \
                    --attack_name cw \
                    --use_last_n_hidden 1\
                    --targeted True \
                    --lr 0.01 \
                    --eps 0.5 \
                    --grad_sparsity 99 \
                    --nb_iter $nb_iter \
                    --confidence $confidence \
                    --norm inf \
                    --binary_search_steps 1 \
                    --initial_const $initial_const
                done
            done
        done
    done
done

rm "./slurm-${SLURM_JOB_ID}.out"
