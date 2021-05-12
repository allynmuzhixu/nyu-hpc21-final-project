#!/bin/bash

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2GB
#SBATCH --job-name=nnls

module purge
module load amber/openmpi/intel/20.06

mpiexec ./nnls