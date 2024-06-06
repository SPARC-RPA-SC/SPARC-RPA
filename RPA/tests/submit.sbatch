#!/bin/bash
#SBATCH -J Si8_RPA
#SBATCH -N 1 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t1:00:00
#SBATCH -oReport-%j.out
cd $SLURM_SUBMIT_DIR

module purge
module load intel
module load intel-mpi

srun ../../lib/rpacalc -name Si8 > out.log
