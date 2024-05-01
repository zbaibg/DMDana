#!/bin/bash
#SBATCH -J lindbladInit
#SBATCH -p pre
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem-per-cpu=4000

module load openmpi/4.1.3-gcc-11.3.0
pwd; hostname; date

echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks"
echo  "with each node getting $SLURM_NTASKS_PER_NODE tasks."

MPICMD="srun --mpi=pmix -n $SLURM_NTASKS"
DIRJ="/software/groups/ping_group/shared/apps/jdftx-1.7.0/build"
DIRF="/software/groups/ping_group/zbai29/jdftx-202209-ru/build-FeynWann/lindbladInit_for-DMD-4.5.6"
${MPICMD} ${DIRF}/init_for-DMD -i lindbladInit.in > lindbladInit.out

