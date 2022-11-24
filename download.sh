# Load modules
module load python/3.10.2 gcc/9.3.0 git-lfs/2.11.0

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_ENV/env
source $TMP_ENV/env/bin/activate
python -m pip install --no-index -U pip setuptools
python -m pip install -U build wheel

# Download package dependencies
mkdir -p $HOME/python_wheels

# build transformers
# This private build is being upstreamed at https://github.com/huggingface/transformers/pull/20305
cd $(mktemp -d)
git clone https://github.com/andreasmadsen/transformers.git
cd transformers
git checkout roberta-prelayernorm
python -m build
cp dist/*-py3-none-any.whl $HOME/python_wheels

# install dependencies
cd $HOME/python_wheels
python -m pip download --no-deps 'huggingface-hub<1.0,>=0.10.0'

# Install project
cd $HOME/workspace/economical-roar
python -m pip install --no-index --find-links $HOME/python_wheels -e .

# Envs
export NO_GCE_CHECK=true
export TF_CPP_MIN_LOG_LEVEL=1

# Fetch dataset
python experiments/download.py --persistent-dir $SCRATCH/ecoroar
