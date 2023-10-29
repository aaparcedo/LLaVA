#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=torchattacks


log_folder="/home/aaparcedo/LLaVA/attacks"
export run_name="deepfool_attack"
export log_file="/${log_folder}/results/${run_name}_${SLURM_JOB_ID}.log"

exec &> $log_file

module load cuda/cuda-12.1
echo "Starting time: $(date)" 

### Activate conda enviroment
source activate /home/aaparcedo/my-envs/llava_clone2
export PYTHONPATH=$PYTHONPATH:/home/aaparcedo/my-envs/llava_clone2/bin/python3.10
export SLURM_JOB_ID=$SLURM_JOB_ID



rm "./slurm-${SLURM_JOB_ID}.out"

# Run python
/home/aaparcedo/my-envs/llava_clone2/bin/python3.10 torchattacks_coco.py

echo "Ending time: $(date)" 


