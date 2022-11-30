#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0"

# V100
#declare -A time=( ["roberta-sb BoolQ"]="0:18:0"  ["roberta-sb CoLA"]="0:17:0"  ["roberta-sb IMDB"]="0:41:0"  ["roberta-sb QQP"]="5:20:0"  ["roberta-sb SST2"]="1:00:0"
#                  ["roberta-sl BoolQ"]="0:31:0"  ["roberta-sl CoLA"]="0:28:0"  ["roberta-sl IMDB"]="1:29:0"  ["roberta-sl QQP"]="10:00:0"  ["roberta-sl SST2"]="1:57:0"
#                  ["roberta-m40 BoolQ"]="0:31:0" ["roberta-m40 CoLA"]="0:28:0" ["roberta-m40 IMDB"]="1:29:0" ["roberta-m40 QQP"]="10:00:0" ["roberta-m40 SST2"]="1:57:0" )

declare -A time=( ["roberta-sb BoolQ"]="0:30:0"  ["roberta-sb CoLA"]="0:30:0"  ["roberta-sb IMDB"]="1:00:0"  ["roberta-sb QQP"]="6:00:0"  ["roberta-sb SST2"]="1:20:0"
                  ["roberta-sl BoolQ"]="0:50:0"  ["roberta-sl CoLA"]="0:50:0"  ["roberta-sl IMDB"]="2:00:0"  ["roberta-sl QQP"]="10:00:0"  ["roberta-sl SST2"]="2:30:0"
                  ["roberta-m40 BoolQ"]="0:50:0" ["roberta-m40 CoLA"]="0:50:0" ["roberta-m40 IMDB"]="2:00:0" ["roberta-m40 QQP"]="10:00:0" ["roberta-m40 SST2"]="2:30:0" )

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
