#!/bin/bash
#SBATCH --job-name=rir_simulation
#SBATCH --output=logs/rir_sim_%j.out
#SBATCH --error=logs/rir_sim_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

matlab -nodisplay -nosplash -r "addpath('src/lib'); addpath('src/audioprocessing/scripts/library/');addpath('src/SMIR-Generator/'); simulated_rir_generator('/mnt/share/carabias/datasets/simulated',0,40,20); exit"