#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8 gcc/9.3.0 cuda/11.2/cudnn/8.1

# Create environment
rm -rf $SLURM_TMPDIR/env
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install project
cd $HOME/workspace/economical-roar
python -m pip install -U pip
python -m pip install -e .

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Run code
unalias py
py () {
    python -u -X faulthandler "$1" "${@:2}" --persistent-dir $SCRATCH/ecoroar
}
cd $HOME/workspace/economical-roar
