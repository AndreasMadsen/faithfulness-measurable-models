#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                     V100                      epochs
declare -A time=( # ["small BoolQ"]="3:10:0" ["large BoolQ"]="7:58:0"  15
                    ["small BoolQ"]="4:00:0" ["large BoolQ"]="9:00:0"
                  # ["small CB"]="?:??:?"    ["large CB"]="?:??:?"     20
                    ["small CB"]="3:00:0"    ["large CB"]="3:00:0"
                  # ["small CoLA"]="1:8:0"   ["large CoLA"]="2:33:0"   15
                    ["small CoLA"]="1:40:0"  ["large CoLA"]="3:00:0"
                  # ["small IMDB"]="1:38:0"  ["large IMDB"]="6:04:0"   10
                    ["small IMDB"]="2:30:0"  ["large IMDB"]="7:00:0"
                  # ["small MNLI"]="4:33:0"  ["large MNLI"]="9:53:0"   10
                    ["small MNLI"]="5:11:0"  ["large MNLI"]="11:00:0"
                  # ["small MRPC"]="0:??:0" ["large MRPC"]="?:??:?"    20
                    ["small MRPC"]="3:00:0"  ["large MRPC"]="3:00:0"
                  # ["small QNLI"]="?:??:?"  ["large QNLI"]="?:??:?"   20
                    ["small QNLI"]="5:00:0"  ["large QNLI"]="11:00:0"
                  # ["small QQP"]="3:01:0"   ["large QQP"]="8:43:0"    10
                    ["small QQP"]="4:00:0"   ["large QQP"]="9:50:0"
                  # ["small RTE"]="?:??:?"   ["large RTE"]="?:??:?"    20
                    ["small RTE"]="3:00:0"   ["large RTE"]="3:00:0"
                  # ["small SST2"]="0:48:0"  ["large SST2"]="1:35:0"   10
                    ["small SST2"]="1:20:0"  ["large SST2"]="2:20:0"
                  # ["small WNLI"]="?:??:?"  ["large WNLI"]="?:??:?"   20
                    ["small WNLI"]="3:00:0"  ["large WNLI"]="3:00:0" )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for max_masking_ratio in 0 100
        do
            for seed in 0 1 2 3 4
            do
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio "${max_masking_ratio}"
            done
        done
    done
done
