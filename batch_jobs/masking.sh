#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                     V100                      epochs
declare -A time=( # ["small BoolQ"]="0:35:0" ["large BoolQ"]="1:16:0"  15
                    ["small BoolQ"]="1:00:0" ["large BoolQ"]="1:50:0"
                  # ["small CB"]="0:21:0"    ["large CB"]="0:17:0"     50
                    ["small CB"]="0:50:0"    ["large CB"]="0:50:0"
                  # ["small CoLA"]="1:08:0"  ["large CoLA"]="2:33:0"   15
                    ["small CoLA"]="1:40:0"  ["large CoLA"]="3:00:0"
                  # ["small IMDB"]="1:44:0"  ["large IMDB"]="4:16:0"   10
                    ["small IMDB"]="2:20:0"  ["large IMDB"]="5:40:0"
                  # ["small MNLI"]="5:20:0"  ["large MNLI"]="10:30:0"  10
                    ["small MNLI"]="6:20:0"  ["large MNLI"]="11:30:0"
                  # ["small MRPC"]="0:11:0"  ["large MRPC"]="0:20:0"   20
                    ["small MRPC"]="0:35:0"  ["large MRPC"]="0:40:0"
                  # ["small QNLI"]="2:52:0"  ["large QNLI"]="6:28:0"   20
                    ["small QNLI"]="3:30:0"  ["large QNLI"]="7:00:0"
                  # ["small QQP"]="3:15:0"   ["large QQP"]="8:05:0"    10
                    ["small QQP"]="4:00:0"   ["large QQP"]="9:00:0"
                  # ["small RTE"]="0:25:0"   ["large RTE"]="0:34:0"    30
                    ["small RTE"]="0:55:0"   ["large RTE"]="0:55:0"
                  # ["small SST2"]="0:49:0"  ["large SST2"]="1:46:0"   10
                    ["small SST2"]="1:20:0"  ["large SST2"]="2:00:0"
                  # ["small WNLI"]="0:07:0"  ["large WNLI"]="0:10:0"   20
                    ["small WNLI"]="0:30:0"  ["large WNLI"]="0:30:0" )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for max_masking_ratio in 0 20 40 60 80 100
        do
            for seed in 0 1 2 3 4
            do
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio "${max_masking_ratio}" \
                    --masking-strategy uni
            done
        done
    done
done

for model in 'roberta-sb' 'roberta-sl'
do
    for dataset in 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for masking_strategy in half-ran half-det
        do
            for seed in 0 1 2 3 4
            do
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio 100 \
                    --masking-strategy "${masking_strategy}"
            done
        done
    done
done
