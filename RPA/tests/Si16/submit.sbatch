#!/bin/bash
#SBATCH -J Si16_RPA_SC
#SBATCH -N 1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t 1:00:00
#SBATCH -o Report-%j.out
cd $SLURM_SUBMIT_DIR

module purge
module load intel
module load intel-mpi

srun ../../lib/rpacalc -name Si16 > out.log
grep -i "subspace iteration" out.log > subtimings.log
