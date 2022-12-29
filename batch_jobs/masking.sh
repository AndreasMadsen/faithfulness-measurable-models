#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="small"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

# V100
#declare -A time=( ["small BoolQ"]="0:15:0" ["small CoLA"]="0:08:0" ["small IMDB"]="0:39:0" ["small QQP"]="1:40:0" ["small SST2"]="0:21:0"
#                  ["large BoolQ"]="0:27:0" ["large CoLA"]="0:13:0" ["large IMDB"]="1:24:0" ["large QQP"]="2:45:0" ["large SST2"]="0:42:0" )
declare -A time=(  ["small BoolQ"]="0:30:0" ["small CoLA"]="0:30:0" ["small IMDB"]="1:15:0" ["small QQP"]="2:50:0" ["small SST2"]="0:50:0"
                   ["large BoolQ"]="0:50:0" ["large CoLA"]="0:40:0" ["large IMDB"]="3:00:0" ["large QQP"]="4:00:0" ["large SST2"]="1:10:0" )

for model in 'roberta-sb' 'roberta-sl' 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CoLA' 'IMDB' 'QQP' 'SST2'
    do
        for max_masking_ratio in {0..100..20}
        do
            submit_seeds "${time[${size[$model]} $dataset]}" "$seeds" $(job_script gpu) \
                experiments/masking.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --max-masking-ratio "${max_masking_ratio}"
        done
    done
done
