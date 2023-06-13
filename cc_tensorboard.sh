# Load modules
module load python/3.10.2 gcc/9.3.0 git-lfs/2.11.0 hdf5/1.12.1

# Create environment
virtualenv --app-data $SCRATCH/virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index -U pip setuptools

# Install tensorboard
python -m pip install --no-index --find-links $HOME/python_wheels 'tensorflow==2.12.0' tensorboard tensorboard_plugin_profile

tb() {
    logdir=
    if [ $# -eq 0 ]; then
        printf >&2 'fatal: provide at least one logdir\n'
    fi
    for filepath; do
        filename=$(basename -- ${filepath})
        logdir="${logdir}${logdir:+,}${filename}:${filepath}"
    done

    echo "Run this command on the client machine:"
    echo ""
    echo "   " ssh -N -L localhost:18343:${SLURMD_NODENAME}:18343 ${USER}@${CC_CLUSTER}.computecanada.ca
    echo ""
    echo 'Then open:'
    echo "   " "http://localhost:18343"
    echo ""
    echo "Starting tensorboard:"
    (set -x; tensorboard --host 0.0.0.0 --port 18343 --logdir_spec "${logdir}")
}
