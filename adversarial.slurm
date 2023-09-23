#!/bin/bash

# Define your variables
experiment_dataset="fgsm_clip"
augmentation_technique="none"
experiment_type="original"
experiment_version="v1"

experiment_metric="clasification"
experiment_name="fgsm_clip_food101_v1"


# Use heredoc to create the job script with the variables
sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --constraint=h100
#SBATCH --job-name=vqa_${experiment_type}_${experiment_name}_${experiment_metric}

#SBATCH --error=vqa_${experiment_type}_${experiment_name}_${experiment_metric}.%J.err
#SBATCH --output=vqa_${experiment_type}_${experiment_name}_${experiment_metric}.%J.out

export job_name="vqa_${experiment_type}_${experiment_name}_${experiment_metric}"
export log_dir="/home/crcvreu.student2/LLaVA/adversarial/job_logs/${experiment_type}_${experiment_name}/\${job_name}-\$SLURM_JOB_ID"
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
conda activate /home/crcvreu.student2/my-envs/llava_cu121

### Run python
/home/crcvreu.student2/my-envs/llava_cu121/bin/python3.10 -m llava.eval.model_vqa --model-name LLAMA-on-LLaVA --question-file dataset/subsets/food101_questions.jsonl --image-folder /home/crcvreu.student2/LLaVA/adversarial/datasets/${experiment_type}_food101_no_labels_v1 --answers-file /home/crcvreu.student2/LLaVA/adversarial/vqa_answers/${experiment_type}_${experiment_name}_${experiment_metric}_\$SLURM_JOB_ID.json --data-augmentation ${augmentation_technique}

sleep 3

echo "Ending time: \$(date)" >> \$benchmark_logs
echo "ulimit -l: " >> \$benchmark_logs
ulimit -l >> \$benchmark_logs

## Directory Cleanup ##
mv \$job_name.\$SLURM_JOB_ID.err \$log_dir
mv \$job_name.\$SLURM_JOB_ID.out \$log_dir
EOF                                                                                                                                                         1,1           Top

