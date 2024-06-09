#!/bin/bash
#SBATCH -J Si24_RPA_SC
#SBATCH -N 4 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t 0:15:00
#SBATCH -o Report-%j.out
cd $SLURM_SUBMIT_DIR

module purge
module load intel
module load intel-mpi

srun ../../lib/rpacalc -name Si24 > out.log
grep -i "subspace iteration" out.log > subtimings.log
