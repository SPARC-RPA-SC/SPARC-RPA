#!/bin/bash
#SBATCH -J Si40_RPA_SC
#SBATCH --account=gts-echow7-coda20
#SBATCH -N 16 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t 0:15:00
#SBATCH -o Report-%j.out
cd $SLURM_SUBMIT_DIR

module purge
module load intel
module load intel-mpi

srun ../../lib/rpacalc -name Si40 > out.log
grep -i "subspace iteration" out.log > subtimings.log
