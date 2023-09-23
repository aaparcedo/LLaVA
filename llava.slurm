#!/bin/bash
##SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --constraint=gmem80
#SBATCH --job-name=llava-13b

#SBATCH --error=llava-13b.%J.err
#SBATCH --output=llava-13b.%J.out

export job_name="llava-13b"
export log_dir="/home/al402390/LLaVA/job_logs/$job_name-$SLURM_JOB_ID"
mkdir -p $log_dir
export debug_logs="$log_dir/job_$SLURM_JOB_ID.log"
export benchmark_logs="$log_dir/job_$SLURM_JOB_ID.log"

module load cuda/11.0

cd $SLURM_SUBMIT_DIR

echo "Slurm working directory: $SLURM_SUBMIT_DIR" >> $debug_logs
echo "JobID: $SLURM_JOB_ID" >> $debug_logs
echo "Running on $SLURM_NODELIST" >> $debug_logs
echo "Running on $SLURM_NNODES nodes." >> $debug_logs
echo "Running on $SLURM_NPROCS processors." >> $debug_logs
echo "Current working directory is `pwd`" >> $debug_logs

echo "Modules loaded:" >> $debug_logs
module list >> $debug_logs
echo "mpirun location: $(which mpirun)" >> $debug_logs

echo "Starting time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

### Running the program ###

## Select python File to run
source activate llava

python -m llava.eval.run_llava --model-name LLAMA-on-LLaVA --image-file /share/datasets/coco2014/coco/trainval2014/COCO_val2014_000000441147.jpg --query "Does one have to be 'tortured' to be a great artist?"

sleep 3

echo "Ending time: $(date)" >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

## Directory Cleanup ##
mv $job_name.$SLURM_JOB_ID.err $log_dir
mv $job_name.$SLURM_JOB_ID.out $log_dir
