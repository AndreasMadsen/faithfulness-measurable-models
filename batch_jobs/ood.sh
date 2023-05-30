#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100
declare -A time=( # ["small bAbI-1"]="0:06"  ["large bAbI-1"]="0:10"
                    ["small bAbI-1"]="0:20"  ["large bAbI-1"]="0:20"
                  # ["small bAbI-2"]="0:12"  ["large bAbI-2"]="0:10"
                    ["small bAbI-2"]="0:25"  ["large bAbI-2"]="0:20"
                  # ["small bAbI-3"]="0:09"  ["large bAbI-3"]="0:09"
                    ["small bAbI-3"]="0:20"  ["large bAbI-3"]="0:20"
                  # ["small BoolQ"]="?:??"   ["large BoolQ"]="?:??"
                    ["small BoolQ"]="0:45"   ["large BoolQ"]="1:30"
                  # ["small CB"]="0:07"      ["large CB"]="0:05"
                    ["small CB"]="0:20"      ["large CB"]="0:20"
                  # ["small CoLA"]="0:09"    ["large CoLA"]="0:10"
                    ["small CoLA"]="0:20"    ["large CoLA"]="0:20"
                  # ["small IMDB"]="0:??"    ["large IMDB"]="?:??"
                    ["small IMDB"]="0:20"    ["large IMDB"]="5:40"
                  # ["small MIMIC-a"]="0:07" ["large MIMIC-a"]="0:05"
                    ["small MIMIC-a"]="0:20" ["large MIMIC-a"]="0:20"
                  # ["small MIMIC-d"]="0:07" ["large MIMIC-d"]="0:07"
                    ["small MIMIC-d"]="0:20" ["large MIMIC-d"]="0:20"
                  # ["small MNLI"]="?:??"    ["large MNLI"]="?:??"
                    ["small MNLI"]="6:20"    ["large MNLI"]="11:30"
                  # ["small MRPC"]="0:06"    ["large MRPC"]="0:06"
                    ["small MRPC"]="0:20"    ["large MRPC"]="0:20"
                  # ["small QNLI"]="?:??"    ["large QNLI"]="?:??"
                    ["small QNLI"]="3:30"    ["large QNLI"]="7:00"
                  # ["small QQP"]="?:??"     ["large QQP"]="?:??"
                    ["small QQP"]="4:00"     ["large QQP"]="9:00"
                  # ["small RTE"]="0:06"     ["large RTE"]="0:07"
                    ["small RTE"]="0:20"     ["large RTE"]="0:20"
                  # ["small SST2"]="?:??"    ["large SST2"]="?:??"
                    ["small SST2"]="1:10"    ["large SST2"]="2:00"
                  # ["small WNLI"]="?:??"    ["large WNLI"]="?:??"
                    ["small WNLI"]="0:30"    ["large WNLI"]="0:30"      )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'MIMIC-a' 'MIMIC-d' 'MRPC' 'RTE' 'SST2'  # 'IMDB' 'MNLI' 'QNLI' 'QQP' 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'inp-grad-abs' 'inp-grad-sign' 'int-grad-abs' 'int-grad-sign' 'loo-sign' 'loo-abs' 'beam-sign-10'
        do
            for max_masking_ratio in 0 100
            do
                for dist_repeat in 1 # 2 4 6 8 10
                do
                    if [[ "${explainer}" == "beam-"* ]]; then
                      if [[ "${dataset}" == "bAbI-3" || "${dataset}" == "MIMIC-a" || "${dataset}" == "MIMIC-d" ]]; then
                        break;
                      fi
                    fi

                    submit_seeds $(multiply_time "${time[${size[$model]} $dataset]}" $dist_repeat) "$seeds" $(job_script gpu) \
                        experiments/ood.py \
                        --model "${model}" \
                        --dataset "${dataset}" \
                        --max-masking-ratio "${max_masking_ratio}" \
                        --masking-strategy 'half-det' \
                        --validation-dataset 'both' \
                        --explainer "${explainer}" \
                        --split 'test' \
                        --ood "masf" \
                        --dist-repeats "${dist_repeat}" \
                        --jit-compile \
                        --save-annotated-datasets
                done
            done
        done
    done
done
