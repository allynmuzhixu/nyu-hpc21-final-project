#!/bin/bash

#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=nnls

module purge
module load amber/openmpi/intel/20.06

mpiexec ./nnls