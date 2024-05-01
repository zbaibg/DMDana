#!/bin/bash
#SBATCH -J jdftx
#SBATCH -p pre
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=4000

module load openmpi/4.1.3-gcc-11.3.0

pwd; hostname; date

echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks"
echo  "with each node getting $SLURM_NTASKS_PER_NODE tasks."

MPICMD="srun --mpi=pmix -n $SLURM_NTASKS"
DIRJ="/software/groups/ping_group/shared/apps/jdftx-1.7.0/build"
#DIRF="....../build-FeynWann"
${MPICMD} ${DIRJ}/jdftx -i scf.in > scf.out
#${MPICMD} ${DIRJ}/jdftx -i totalE.in > totalE.out
#${MPICMD} ${DIRJ}/jdftx -i bandstruct.in > bandstruct.out
#${MPICMD} ${DIRJ}/phonon -i phonon.in > phonon.out

