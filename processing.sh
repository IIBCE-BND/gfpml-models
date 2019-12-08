#!/bin/bash
#SBATCH --job-name=gfpml-processing
#SBATCH --nodes 5
#SBATCH --ntasks=5
#SBATCH --mem=2048
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=psoto23.ps@gmail.com

source /etc/profile.d/modules.sh

source /clusteruy/home/fpazos/venv/bin/activate

RUNPATH=/clusteruy/home/fpazos/gfpml-models/
cd $RUNPATH

srun -N 1 -n 1 python processing.py celegans &
srun -N 1 -n 1 python processing.py dmel &
srun -N 1 -n 1 python processing.py hg &
srun -N 1 -n 1 python processing.py mm &
wait