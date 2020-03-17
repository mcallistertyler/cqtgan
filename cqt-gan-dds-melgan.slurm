#!/bin/sh
#SBATCH --partition=EPIC
#SBATCH --account=ie-idi
#SBATCH --time=19:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --job-name="CQTGANDDS"
#SBATCH --output=dds-cqtgan-test-srun.out
#SBATCH --mail-user=tylerm@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

rm woops.txt
rm lastruntime.txt
module purge
module load fosscuda/2019b
module load Python/3.7.4
python --version
source ./cqtgan-libs/bin/activate
source ./set_env.sh 0
date > lastruntime.txt 
#python scripts/train.py --load_path logs/baseline --save_path logs/baseline --data_path cqtdatasets/
python scripts/train.py --load_path logs/baseline --save_path logs/baseline --data_path cqtdatasets/dds/
echo if i reached here something either went right or very wrong > woops.txt 
