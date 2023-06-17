#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100                        epochs
declare -A time=( # ["small bAbI-1"]="0:32"  ["large bAbI-1"]="1:04"   20
                    ["small bAbI-1"]="0:50"  ["large bAbI-1"]="1:40"
                  # ["small bAbI-2"]="0:53"  ["large bAbI-2"]="2:05"   20
                    ["small bAbI-2"]="1:20"  ["large bAbI-2"]="2:40"
                  # ["small bAbI-3"]="1:47"  ["large bAbI-3"]="?:??"   20
                    ["small bAbI-3"]="2:10"  ["large bAbI-3"]="5:00"
                  # ["small BoolQ"]="0:54"   ["large BoolQ"]="2:06"    15
                    ["small BoolQ"]="1:20"   ["large BoolQ"]="2:40"
                  # ["small CB"]="0:13"      ["large CB"]="0:18"       50
                    ["small CB"]="0:30"      ["large CB"]="0:40"
                  # ["small CoLA"]="0:22"    ["large CoLA"]="0:37"     15
                    ["small CoLA"]="2:50"    ["large CoLA"]="1:00"
                  # ["small IMDB"]="1:53"    ["large IMDB"]="4:11"     10
                    ["small IMDB"]="3:00"    ["large IMDB"]="7:00"
                  # ["small MIMIC-a"]="0:52" ["large MIMIC-a"]="2:10"  20
                    ["small MIMIC-a"]="1:20" ["large MIMIC-a"]="2:40"
                  # ["small MIMIC-d"]="1:35" ["large MIMIC-d"]="3:58"  20
                    ["small MIMIC-d"]="2:00" ["large MIMIC-d"]="4:30"
                  # ["small MNLI"]="6:21"    ["large MNLI"]="??:??"    10
                    ["small MNLI"]="8:00"    ["large MNLI"]="16:00"
                  # ["small MRPC"]="0:17"    ["large MRPC"]="0:30"     20
                    ["small MRPC"]="0:40"    ["large MRPC"]="1:00"
                  # ["small QNLI"]="5:05"    ["large QNLI"]="?:??"     20
                    ["small QNLI"]="5:40"    ["large QNLI"]="13:00"
                  # ["small QQP"]="4:39"     ["large QQP"]="?:??"      10
                    ["small QQP"]="6:00"     ["large QQP"]="13:00"
                  # ["small RTE"]="0:23"     ["large RTE"]="0:46"      30
                    ["small RTE"]="0:40"     ["large RTE"]="1:20"
                  # ["small SNLI"]="5:19"    ["large SNLI"]="10:46"    10
                    ["small SNLI"]="5:50"    ["large SNLI"]="11:20"
                  # ["small SST2"]="1:24"    ["large SST2"]="2:42"     10
                    ["small SST2"]="1:50"    ["large SST2"]="3:10"
                  # ["small WNLI"]="0:08"    ["large WNLI"]="0:11"     20
                    ["small WNLI"]="0:30"    ["large WNLI"]="0:30"  )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'MIMIC-a' 'MIMIC-d' 'MRPC' 'RTE' 'SST2' 'SNLI' 'IMDB' 'MNLI' 'QNLI' 'QQP' # 'WNLI'
    do
        for validation_dataset in 'nomask' 'mask' 'both'
        do
            for seed in 0 1 2 3 4
            do
                for masking_strategy in 'uni' 'half-det'
                do
                    submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                        experiments/masking.py \
                        --model "${model}" \
                        --dataset "${dataset}" \
                        --max-masking-ratio 100 \
                        --masking-strategy "${masking_strategy}" \
                        --validation-dataset "${validation_dataset}"
                done

                # baseline (as in max-masking-ratio == 0)
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio 0 \
                    --masking-strategy 'half-det' \
                    --validation-dataset "${validation_dataset}"
            done
        done
    done
done

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    break
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'MIMIC-a' 'MIMIC-d' 'MRPC' 'RTE' 'SST2' # 'SNLI' 'IMDB' 'MNLI' 'QNLI' 'QQP' # 'WNLI'
    do
        for max_masking_ratio in 20 40 60 80
        do
           for seed in 0 1 2 3 4
            do
                submit_seeds "${time[${size[$model]} $dataset]}" "$seed" $(job_script gpu) \
                    experiments/masking.py \
                    --model "${model}" \
                    --dataset "${dataset}" \
                    --max-masking-ratio "${max_masking_ratio}" \
                    --masking-strategy 'half-det' \
                    --validation-dataset 'both'
            done
        done
    done
done
