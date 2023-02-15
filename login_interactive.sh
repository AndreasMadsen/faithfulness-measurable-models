# Load modules
module load python/3.10.2 gcc/9.3.0 git-lfs/2.11.0 hdf5/1.12.1

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_ENV/env
source $TMP_ENV/env/bin/activate
python -m pip install --no-index -U pip setuptools wheel

# Envs
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index --find-links $HOME/python_wheels -e .
