#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.10.2 gcc/9.3.0 git-lfs/2.11.0 cuda/11.2 cudnn/8.2.0 hdf5/1.12.1

# Create environment
virtualenv --app-data $SCRATCH/virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index -U pip setuptools wheel

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Run code
cd $HOME/workspace/economical-roar

if [ -z "${RUN_SEEDS}" ]; then
    python -u -X faulthandler "$1" "${@:2}" --persistent-dir $SCRATCH/ecoroar
else
    for seed in $(echo $RUN_SEEDS)
    do
        echo Running $seed
        python -u -X faulthandler "$1" "${@:2}" --seed "$seed" --persistent-dir $SCRATCH/ecoroar
    done
fi
