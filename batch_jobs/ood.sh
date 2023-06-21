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
                  # ["small IMDB"]="0:18"    ["large IMDB"]="0:48"
                    ["small IMDB"]="0:30"    ["large IMDB"]="1:20"
                  # ["small MIMIC-a"]="0:07" ["large MIMIC-a"]="0:05"
                    ["small MIMIC-a"]="0:20" ["large MIMIC-a"]="0:20"
                  # ["small MIMIC-d"]="0:07" ["large MIMIC-d"]="0:07"
                    ["small MIMIC-d"]="0:20" ["large MIMIC-d"]="0:20"
                  # ["small MNLI"]="0:21"    ["large MNLI"]="1:02"
                    ["small MNLI"]="1:00"    ["large MNLI"]="1:30"
                  # ["small MRPC"]="0:06"    ["large MRPC"]="0:06"
                    ["small MRPC"]="0:20"    ["large MRPC"]="0:20"
                  # ["small QNLI"]="0:06"    ["large QNLI"]="0:13"
                    ["small QNLI"]="0:30"    ["large QNLI"]="0:40"
                  # ["small QQP"]="0:49"     ["large QQP"]="2:12"
                    ["small QQP"]="1:20"     ["large QQP"]="3:00"
                  # ["small RTE"]="0:06"     ["large RTE"]="0:07"
                    ["small RTE"]="0:20"     ["large RTE"]="0:20"
                  # ["small SST2"]="?:??"    ["large SST2"]="?:??"
                    ["small SST2"]="1:10"    ["large SST2"]="2:00"
                  # ["small SNLI"]="?:??"    ["large SNLI"]="?:??"
                    ["small SNLI"]="0:20"    ["large SNLI"]="0:20"
                  # ["small WNLI"]="?:??"    ["large WNLI"]="?:??"
                    ["small WNLI"]="0:30"    ["large WNLI"]="0:30"      )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'MIMIC-a' 'MIMIC-d' 'MRPC' 'RTE' 'SST2' 'SNLI' 'IMDB' 'MNLI' 'QNLI' 'QQP' # 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'inp-grad-abs' 'inp-grad-sign' 'int-grad-abs' 'int-grad-sign' 'loo-sign' 'loo-abs' 'beam-sign-10'
        do
            for max_masking_ratio in 0 100
            do
                for dist_repeat in 1 # 2 4 6 8 10
                do
                    if [[ "${explainer}" == "beam-"* ]]; then
                      if [[ "${dataset}" == "bAbI-3" || "${dataset}" == "MIMIC-"* || "${dataset}" == "IMDB" ]]; then
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
