#!/bin/bash
#SBATCH --job-name=CVAE-SIM-reliability
#SBATCH --output=slurm-01-CVAE-SIM-reliability-outputs.txt
#SBATCH --error=slurm-01-CVAE-SIM-reliability-errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpuv100

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
notebook_name='11-AA-train-cvae-reliability.ipynb'
for i in {21..40}
do
    save_dir=../reliability-results/good_init_${i}_
    outname=$save_dir$notebook_name
    date
    echo $save_dir
    echo $outname
    papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p save_dir $save_dir
done

