# Load modules
# Tensorflow is hardcoded to cuda/11.8 and cudnn/8.6 but load them when they become offically avilable anyway
module load python/3.10.2 gcc/9.3.0 git-lfs/2.11.0 hdf5/1.12.1 arrow/10.0.1

# Create environment
virtualenv --app-data $SCRATCH/virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index -U pip setuptools

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index --find-links $HOME/python_wheels -e .

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
