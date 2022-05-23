#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# V100     time=( ["BoolQ"]="0:11:0" ["CoLA"]="0:08:0" ["IMDB"]="0:28:0" ["QQP"]="1:47:0" ["SST2"]="0:22:0" )
declare -A time=( ["BoolQ"]="0:20:0" ["CoLA"]="0:20:0" ["IMDB"]="0:45:0" ["QQP"]="2:10:0" ["SST2"]="0:35:0" )

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
