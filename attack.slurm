#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=attack

#SBATCH --error=attack.%J.err
#SBATCH --output=attack.%J.out

export job_name="attack"
export log_dir="/home/crcvreu.student2/LLaVA-13B/job_logs/llava_robust/\${job_name}-\$SLURM_JOB_ID"
mkdir -p \$log_dir
export debug_logs="\$log_dir/job_\$SLURM_JOB_ID.log"
export benchmark_logs="\$log_dir/job_\$SLURM_JOB_ID.log"

module load anaconda/anaconda3
module load cuda/cuda-12.1

cd \$SLURM_SUBMIT_DIR

echo "Slurm working directory: \$SLURM_SUBMIT_DIR" >> \$debug_logs
echo "JobID: \$SLURM_JOB_ID" >> \$debug_logs
echo "Running on \$SLURM_NODELIST" >> \$debug_logs
echo "Running on \$SLURM_NNODES nodes." >> \$debug_logs
echo "Running on \$SLURM_NPROCS processors." >> \$debug_logs
echo "Current working directory is \`pwd\`" >> \$debug_logs
echo "Running on 2 80G GPU(S)" >> \$debug_logs

echo "Modules loaded:" >> \$debug_logs
module list >> \$debug_logs

echo "Starting time: \$(date)" >> \$benchmark_logs
echo "ulimit -l: " >> \$benchmark_logs
ulimit -l >> \$benchmark_logs

### Activate conda enviroment
conda activate /home/crcvreu.student2/my-envs/llava_clone2

### Run python
/home/crcvreu.student2/my-envs/llava_clone2/bin/python3.10 pgd_3.py --set 0 --set_num 25000 --save_image True --descriptors True --image_list imagenet_test.txt

sleep 3

echo "Ending time: \$(date)" >> \$benchmark_logs
echo "ulimit -l: " >> \$benchmark_logs
ulimit -l >> \$benchmark_logs

## Directory Cleanup ##
mv \$job_name.\$SLURM_JOB_ID.err \$log_dir
mv \$job_name.\$SLURM_JOB_ID.out \$log_dir
