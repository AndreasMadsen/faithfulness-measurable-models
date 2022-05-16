#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

# Actual   time="0:??:0"
declare -r time="0:22:0"

for max_masking_ratio in {0..100..5}
do
    submit_seeds ${time} "$seeds" "masking-effect/imdb_s-%s_m-${max_masking_ratio}.json" \
        $(job_script gpu) \
        experiments/imdb_masking.py \
        --max-masking-ratio "${max_masking_ratio}"
done
