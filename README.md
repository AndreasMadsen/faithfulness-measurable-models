# Ecconomical ROAR

## Install

This module is not published on PyPi but you can install directly with:

```bash
python -m pip install -e .
```

## Experiments

### Tasks

There are scripts for each dataset:

* IMDB: `python experiments/imdb_masking.py`

### Parameters

Each of the above scripts takes the same set of CLI arguments. You can learn
about each argument with `--help`. The most important arguments which
will allow you to run the experiments presented in the paper are:

* `--max-masking-ratio`: The maximum masking ratio to apply on the training dataset.

## Running on a HPC setup

For downloading content we provide a `download.sh` script.

Additionally, we provide scripts for submitting all jobs to a Slurm
queue, in `batch_jobs/`.
The jobs automatically use `$SCRATCH/ecoroar` as the persistent dir.
