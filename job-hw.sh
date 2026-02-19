#!/bin/bash
#SBATCH --job-name=Lab2
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adam.jacobs@ufl.edu
#SBATCH --account=eel6763
#SBATCH --qos=eel6763
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000mb
#SBATCH -t 00:05:00	
#SBATCH -o OutputR32N1000
#SBATCH -e myerr
srun --mpi=$HPC_PMIX ./main 1000
