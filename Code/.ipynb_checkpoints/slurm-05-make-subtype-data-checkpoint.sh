#!/bin/bash
#SBATCH --job-name=CVAE-make-data
#SBATCH --output=slurm-05-CVAE-CVAE-make-data-outputs.txt
#SBATCH --error=slurm-05-CVAE-make-data-errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=32gb


bash

#source .bash_profile


notebook_name=08-gen-datsa-subtypes
outname=08-gen-datsa-subtypes_2
papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 60 --progress-bar

