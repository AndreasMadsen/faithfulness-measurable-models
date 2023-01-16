#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="small"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

# V100
#                   20 epochs,              20 epochs,              3 epochs,               3 epochs,              3 epochs
#declare -A time=( ["small BoolQ"]="3:20:0" ["small CoLA"]="1:12:0" ["small IMDB"]="0:39:0" ["small QQP"]="1:40:0" ["small SST2"]="0:21:0"
#                  ["large BoolQ"]="8:20:0" ["large CoLA"]="2:40:0" ["large IMDB"]="1:24:0" ["large QQP"]="2:45:0" ["large SST2"]="0:42:0" )
declare -A time=(  ["small BoolQ"]="4:00:0" ["small CoLA"]="1:40:0" ["small IMDB"]="1:15:0" ["small QQP"]="2:50:0" ["small SST2"]="0:50:0"
                   ["large BoolQ"]="9:00:0" ["large CoLA"]="3:20:0" ["large IMDB"]="3:00:0" ["large QQP"]="4:00:0" ["large SST2"]="1:10:0" )

for model in 'roberta-sb' 'roberta-sl' 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CoLA'
    do
        for max_masking_ratio in {0..100..20}
        do
            submit_seeds "${time[${size[$model]} $dataset]}" "$seeds" $(job_script gpu) \
                experiments/masking.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --max-masking-ratio "${max_masking_ratio}" \
                --max-epochs 20
        done
    done
done
