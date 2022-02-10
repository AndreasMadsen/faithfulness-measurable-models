#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00

# Load modules
module load python/3.8.10 gcc/9.3.0 cuda/11.4

# Create environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index -U pip
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Enable offline model
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_GCE_CHECK=true

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