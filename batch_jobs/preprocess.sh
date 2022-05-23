#!/bin/bash
source "batch_jobs/_job_script.sh"

sbatch --time=4:00:00 -J "preprocess" \
    -o "$SCRATCH"/ecoroar/logs/%x.%j.out -e "$SCRATCH"/ecoroar/logs/%x.%j.err \
    $(job_script cpu) \
    experiments/preprocess.py
