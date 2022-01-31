#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8.10 gcc/9.3.0 cuda/11.4

# Create environment
rm -rf $SLURM_TMPDIR/env
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index -U pip
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Run code
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_GCE_CHECK=true

cd $HOME/workspace/economical-roar
