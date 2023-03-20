#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100                        epochs
declare -A time=( # ["small bAbI-1"]="0:21"  ["large bAbI-1"]="0:44"   20
                    ["small bAbI-1"]="0:40"  ["large bAbI-1"]="1:00"
                  # ["small bAbI-2"]="0:33"  ["large bAbI-2"]="1:14"   20
                    ["small bAbI-2"]="0:50"  ["large bAbI-2"]="1:30"
                  # ["small bAbI-3"]="1:02"  ["large bAbI-3"]="2:29"   20
                    ["small bAbI-3"]="1:20"  ["large bAbI-3"]="3:00"
                  # ["small BoolQ"]="0:35"   ["large BoolQ"]="1:16"    15
                    ["small BoolQ"]="1:00"   ["large BoolQ"]="1:50"
                  # ["small CB"]="0:21"      ["large CB"]="0:17"       50
                    ["small CB"]="0:50"      ["large CB"]="0:50"
                  # ["small CoLA"]="1:08"    ["large CoLA"]="2:33"     15
                    ["small CoLA"]="1:40"    ["large CoLA"]="3:00"
                  # ["small IMDB"]="1:44"    ["large IMDB"]="4:16"     10
                    ["small IMDB"]="2:20"    ["large IMDB"]="5:40"
                  # ["small MIMIC-a"]="0:38" ["large MIMIC-a"]="1:26"  20
                    ["small MIMIC-a"]="1:00" ["large MIMIC-a"]="1:50"
                  # ["small MIMIC-d"]="1:02" ["large MIMIC-d"]="2:32"  20
                    ["small MIMIC-d"]="1:30" ["large MIMIC-d"]="3:00"
                  # ["small MNLI"]="5:20"    ["large MNLI"]="10:30"    10
                    ["small MNLI"]="6:20"    ["large MNLI"]="11:30"
                  # ["small MRPC"]="0:11"    ["large MRPC"]="0:20"     20
                    ["small MRPC"]="0:35"    ["large MRPC"]="0:40"
                  # ["small QNLI"]="2:52"    ["large QNLI"]="6:28"     20
                    ["small QNLI"]="3:30"    ["large QNLI"]="7:00"
                  # ["small QQP"]="3:15"     ["large QQP"]="8:05"      10
                    ["small QQP"]="4:00"     ["large QQP"]="9:00"
                  # ["small RTE"]="0:25"     ["large RTE"]="0:34"      30
                    ["small RTE"]="0:55"     ["large RTE"]="0:55"
                  # ["small SST2"]="0:49"    ["large SST2"]="1:46"     10
                    ["small SST2"]="1:20"    ["large SST2"]="2:00"
                  # ["small WNLI"]="0:07"    ["large WNLI"]="0:10"     20
                    ["small WNLI"]="0:30"    ["large WNLI"]="0:30"  )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MIMIC-a' 'MIMIC-d' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
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
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MIMIC-a' 'MIMIC-d' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
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
