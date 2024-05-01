#!/bin/bash
#SBATCH -J dm
#SBATCH -p pre
#SBATCH -t 24:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=59
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4G

module load openmpi/4.1.3-gcc-11.3.0

pwd; hostname; date

echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks"
echo  "with each node getting $SLURM_NTASKS_PER_NODE tasks."

MPICMD="srun --mpi=pmix -n $SLURM_NTASKS"
DIRDM="/software/groups/ping_group/zbai29/DMDdebug/build"
${MPICMD} ${DIRDM}/DMD  >> out
