# Load modules
module load python/3.8.10 gcc/9.3.0 arrow/5.0.0

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --no-download $TMP_ENV/env
source $TMP_ENV/env/bin/activate

# Download package dependencies
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
pip3 download --no-deps datasets transformers

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index -U pip
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Fetch dataset
python experiments/download.py --persistent-dir $SCRATCH/ecoroar
