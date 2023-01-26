#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

# V100
#declare -A time=( ["small BoolQ"]="3:20:0" ["small CoLA"]="1:12:0" ["small IMDB"]="2:49:0" ["small QQP"]="?:??:0" ["small SST2"]="1:32:0"  ["small MNLI"]="9:11:0"
#                  ["large BoolQ"]="8:20:0" ["large CoLA"]="2:40:0" ["large IMDB"]="7:10:0" ["large QQP"]="19:50:0" ["large SST2"]="3:07:0"  ["large MNLI"]="?:??:0"
declare -A time=(  ["small BoolQ"]="4:00:0" ["small CoLA"]="1:40:0" ["small IMDB"]="3:30:0" ["small QQP"]="21:00:0" ["small SST2"]="2:00:0" ["small MNLI"]="9:11:0"
                   ["large BoolQ"]="9:00:0" ["large CoLA"]="3:20:0" ["large IMDB"]="8:00:0" ["large QQP"]="21:00:0" ["large SST2"]="3:40:0" ["large MNLI"]="21:00:0" )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CoLA' 'SST2' 'IMDB' 'QQP' 'MNLI'
    do
        for max_masking_ratio in 0 100
        do
            for seed in 0 1 2 3 4
            do
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio "${max_masking_ratio}" \
                    --max-epochs 20
            done
        done
    done
done
