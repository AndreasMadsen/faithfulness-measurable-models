#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   time="0:??:0"
declare -A time=( ["IMDB"]="0:35:0" ["MNLI"]="0:50:0" )

for dataset in 'BoolQ' 'CoLA' 'IMDB' 'QQP' 'SST2'
do
    for max_masking_ratio in {0..100..5}
    do
        submit_seeds "${time[$dataset]}" "$seeds" $(job_script gpu) \
            experiments/masking.py \
            --dataset "${dataset}" \
            --max-masking-ratio "${max_masking_ratio}"
    done
done
