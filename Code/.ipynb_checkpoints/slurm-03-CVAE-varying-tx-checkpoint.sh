#!/bin/bash
#SBATCH --job-name=CVAE-SIM-varyTx
#SBATCH --output=slurm-03-varyTx-outputs.txt
#SBATCH --error=slurm-03-varyTx-errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64gb
#SBATCH --partition=anzellos

echo $0

bash

source .bash_profile

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data/aglinska/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data/aglinska/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/data/aglinska/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/data/aglinska/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate tf231

LD_LIBRARY_PATH=/usr/public/tensorflow/2.3.1/lib:/cm/shared/apps/cudnn7.6-cuda10.2/7.6.5.32/lib64:/cm/shared/apps/cuda10.1/toolkit/10.1.243/extras/CUPTI/lib64:/cm/local/apps/cuda/libs/current/lib64:/cm/shared/apps/cuda10.1/toolkit/10.1.243/targets/x86_64-linux/lib:/cm/local/apps/gcc/9.2.0/lib:/cm/local/apps/gcc/9.2.0/lib64:/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64

echo 'running loop'
notebook_name='15-AA-train-cvae-vary-tx'

for jobid in {0..5}
do
    date
    echo 'jobid is ' $jobid
    outname=$notebook_name$jobid
    papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 60 --progress-bar -p job_id $jobid
done

