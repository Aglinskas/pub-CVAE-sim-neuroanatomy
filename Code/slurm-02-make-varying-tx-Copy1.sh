#!/bin/bash
#SBATCH --job-name=CVAE-make-vary-tx
#SBATCH --output=slurm-02-CVAE-make-vary-tx-outputs.txt
#SBATCH --error=slurm-02-CVAE-make-vary-tx-errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=64gb
#SBATCH --array=3-25

echo $0

bash

source .bash_profile


#$SLURM_ARRAY_TASK_ID


notebook_name='14-cvae-make-vary-tx'
outname=$notebook_name$SLURM_ARRAY_TASK_ID

papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 60 --progress-bar -p job_id $save_dir $SLURM_ARRAY_TASK_ID



