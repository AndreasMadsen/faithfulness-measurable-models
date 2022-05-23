# Load modules
module load python/3.8.10 gcc/9.3.0

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_ENV/env
source $TMP_ENV/env/bin/activate

# Download package dependencies
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
python -m pip download --no-deps 'transformers>=4.19.1'
python -m pip download --no-deps 'tensorflow-datasets>=4.5.0'

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index -U pip
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Envs
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Fetch dataset
python experiments/download.py --persistent-dir $SCRATCH/ecoroar
