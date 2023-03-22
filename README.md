# Ecconomical ROAR

## Install

This module is not published on PyPi but you can install directly with:

```bash
python -m pip install -e .
```

## Experiments

### Tasks

There are scripts for each type of experiment:

1. Masked model training: `python experiments/masking.py`
2. Faithfulness evaluation: `python experiments/faithfulness.py`
3. OOD evaluation: `python experiments/ood.py`

### Parameters

Each of the above scripts takes the same set of CLI arguments. You can learn
about each argument with `--help`. The most important arguments which
will allow you to run the experiments presented in the paper are:

* `--dataset`: The dataset used.
* `--model`: The model used.
* `--explainer`: The importance measure used.
* `--masking-strategy` The masking strategy to use during fine-tuning.

## Running on a HPC setup

For downloading the required resources we provide a `experiment/download.py` script.
Additionally, there is a `experiment/preprocess.py` script.

Finally, we provide scripts for submitting all jobs to a Slurm
queue, in `batch_jobs/`. The jobs automatically use `$SCRATCH/ecoroar`
as the persistent dir.

## MIMIC

See https://mimic.physionet.org/gettingstarted/access/ for how to access MIMIC-III.
You will need to download `DIAGNOSES_ICD.csv.gz` and `NOTEEVENTS.csv.gz` and
place them in `mimic/` relative to your presistent dir (e.g. `$SCRATCH/ecoroar/mimic/`).
