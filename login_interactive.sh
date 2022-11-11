# Load modules
module load python/3.8.10 gcc/9.3.0

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --no-download $TMP_ENV/env
source $TMP_ENV/env/bin/activate

# Envs
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index -U pip
python -m pip install --no-index --find-links $HOME/python_wheels -e .
