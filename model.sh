#!/bin/bash
#SBATCH --job-name=gfpml-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=4096
#SBATCH --time=1:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=psoto23.ps@gmail.com

source /etc/profile.d/modules.sh

source /clusteruy/home/fpazos/venv/bin/activate

RUNPATH=/clusteruy/home/fpazos/gfpml-models/
cd $RUNPATH

srun -N 1 -n 1 python model.py celegans biological_process &
# srun -N 1 -n 1 python model.py celegans cellular_component &
# srun -N 1 -n 1 python model.py celegans molecular_function &

# srun -N 1 -n 1 python model.py dmel biological_process &
# srun -N 1 -n 1 python model.py dmel cellular_component &
# srun -N 1 -n 1 python model.py dmel molecular_function &

# srun -N 1 -n 1 python model.py hg biological_process &
# srun -N 1 -n 1 python model.py hg cellular_component &
# srun -N 1 -n 1 python model.py hg molecular_function &

# srun -N 1 -n 1 python model.py mm biological_process &
# srun -N 1 -n 1 python model.py mm cellular_component &
# srun -N 1 -n 1 python model.py mm molecular_function &

# srun -N 1 -n 1 python model.py scer biological_process &
# srun -N 1 -n 1 python model.py scer cellular_component &
# srun -N 1 -n 1 python model.py scer molecular_function &

wait