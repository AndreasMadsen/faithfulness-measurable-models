#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0"

declare -A time=( ["roberta-sb BoolQ"]="2:20:0"  ["roberta-sb CoLA"]="2:20:0"  ["roberta-sb IMDB"]="2:45:0"  ["roberta-sb QQP"]="4:10:0"  ["roberta-sb SST2"]="2:35:0"
                  ["roberta-sl BoolQ"]="2:20:0"  ["roberta-sb CoLA"]="2:20:0"  ["roberta-sb IMDB"]="2:45:0"  ["roberta-sb QQP"]="5:10:0"  ["roberta-sb SST2"]="2:35:0"
                  ["roberta-m40 BoolQ"]="2:20:0" ["roberta-m40 CoLA"]="2:20:0" ["roberta-m40 IMDB"]="2:45:0" ["roberta-m40 QQP"]="5:10:0" ["roberta-m40 SST2"]="2:35:0" )

for model in 'roberta-sb' 'roberta-sl' 'roberta-m40'
do
    for dataset in 'BoolQ' 'CoLA' 'IMDB' 'QQP' 'SST2'
    do
        for max_masking_ratio in {0..100..5}
        do
            submit_seeds "${time[$model $dataset]}" "$seeds" $(job_script gpu) \
                experiments/masking.py \
                --model "${model}" \
                --dataset "${dataset}" \
                --max-masking-ratio "${max_masking_ratio}"
        done
    done
done
