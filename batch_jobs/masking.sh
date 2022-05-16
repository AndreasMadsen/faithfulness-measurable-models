#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   time="0:??:0"
declare -A time=( ["IMDB"]="0:22:0" ["MultiNLI"]="0:30:0")

for dataset in 'IMDB' 'MultiNLP'
    for max_masking_ratio in {0..100..5}
    do
        submit_seeds "${time[$dataset]}" "$seeds" $(job_script gpu) \
            experiments/masking.py \
            --dataset "${dataset}" \
            --max-masking-ratio "${max_masking_ratio}"
    done
done
