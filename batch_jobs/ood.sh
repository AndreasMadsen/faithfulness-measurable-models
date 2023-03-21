#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100
declare -A time=( # ["small test bAbI-1"]="0:21"  ["large test bAbI-1"]="0:44"  ["small train bAbI-1"]="0:21"  ["large train bAbI-1"]="0:44"
                    ["small test bAbI-1"]="0:40"  ["large test bAbI-1"]="1:00"  ["small train bAbI-1"]="0:40"  ["large train bAbI-1"]="1:00"
                  # ["small test bAbI-2"]="0:33"  ["large test bAbI-2"]="1:14"  ["small train bAbI-2"]="0:33"  ["large train bAbI-2"]="1:14"
                    ["small test bAbI-2"]="0:50"  ["large test bAbI-2"]="1:30"  ["small train bAbI-2"]="0:50"  ["large train bAbI-2"]="1:30"
                  # ["small test bAbI-3"]="1:02"  ["large test bAbI-3"]="2:29"  ["small train bAbI-3"]="1:02"  ["large train bAbI-3"]="2:29"
                    ["small test bAbI-3"]="1:20"  ["large test bAbI-3"]="3:00"  ["small train bAbI-3"]="1:20"  ["large train bAbI-3"]="3:00"
                  # ["small test BoolQ"]="0:35"   ["large test BoolQ"]="1:16"   ["small train BoolQ"]="0:35"   ["large train BoolQ"]="1:16"
                    ["small test BoolQ"]="1:00"   ["large test BoolQ"]="1:50"   ["small train BoolQ"]="1:00"   ["large train BoolQ"]="1:50"
                  # ["small test CB"]="0:21"      ["large test CB"]="0:17"      ["small train CB"]="0:21"      ["large train CB"]="0:17"
                    ["small test CB"]="0:50"      ["large test CB"]="0:50"      ["small train CB"]="0:50"      ["large train CB"]="0:50"
                  # ["small test CoLA"]="1:08"    ["large test CoLA"]="2:33"    ["small train CoLA"]="1:08"    ["large train CoLA"]="2:33"
                    ["small test CoLA"]="1:40"    ["large test CoLA"]="3:00"    ["small train CoLA"]="1:40"    ["large train CoLA"]="3:00"
                  # ["small test IMDB"]="1:44"    ["large test IMDB"]="4:16"    ["small train IMDB"]="1:44"    ["large train IMDB"]="4:16"
                    ["small test IMDB"]="2:20"    ["large test IMDB"]="5:40"    ["small train IMDB"]="2:20"    ["large train IMDB"]="5:40"
                  # ["small test MIMIC-a"]="0:38" ["large test MIMIC-a"]="1:26" ["small train MIMIC-a"]="0:38" ["large train MIMIC-a"]="1:26"
                    ["small test MIMIC-a"]="1:00" ["large test MIMIC-a"]="1:50" ["small train MIMIC-a"]="1:00" ["large train MIMIC-a"]="1:50"
                  # ["small test MIMIC-d"]="1:02" ["large test MIMIC-d"]="2:32" ["small train MIMIC-d"]="1:02" ["large train MIMIC-d"]="2:32"
                    ["small test MIMIC-d"]="1:30" ["large test MIMIC-d"]="3:00" ["small train MIMIC-d"]="1:30" ["large train MIMIC-d"]="3:00"
                  # ["small test MNLI"]="5:20"    ["large test MNLI"]="10:30"   ["small train MNLI"]="5:20"    ["large train MNLI"]="10:30"
                    ["small test MNLI"]="6:20"    ["large test MNLI"]="11:30"   ["small train MNLI"]="6:20"    ["large train MNLI"]="11:30"
                  # ["small test MRPC"]="0:11"    ["large test MRPC"]="0:20"    ["small train MRPC"]="0:11"    ["large train MRPC"]="0:20"
                    ["small test MRPC"]="0:35"    ["large test MRPC"]="0:40"    ["small train MRPC"]="0:35"    ["large train MRPC"]="0:40"
                  # ["small test QNLI"]="2:52"    ["large test QNLI"]="6:28"    ["small train QNLI"]="2:52"    ["large train QNLI"]="6:28"
                    ["small test QNLI"]="3:30"    ["large test QNLI"]="7:00"    ["small train QNLI"]="3:30"    ["large train QNLI"]="7:00"
                  # ["small test QQP"]="3:15"     ["large test QQP"]="8:05"     ["small train QQP"]="3:15"     ["large train QQP"]="8:05"
                    ["small test QQP"]="4:00"     ["large test QQP"]="9:00"     ["small train QQP"]="4:00"     ["large train QQP"]="9:00"
                  # ["small test RTE"]="0:25"     ["large test RTE"]="0:34"     ["small train RTE"]="0:25"     ["large train RTE"]="0:34"
                    ["small test RTE"]="0:55"     ["large test RTE"]="0:55"     ["small train RTE"]="0:55"     ["large train RTE"]="0:55"
                  # ["small test SST2"]="0:49"    ["large test SST2"]="1:46"    ["small train SST2"]="0:49"    ["large train SST2"]="1:46"
                    ["small test SST2"]="1:20"    ["large test SST2"]="2:00"    ["small train SST2"]="1:20"    ["large train SST2"]="2:00"
                  # ["small test WNLI"]="0:07"    ["large test WNLI"]="0:10"    ["small train WNLI"]="0:07"    ["large train WNLI"]="0:10"
                    ["small test WNLI"]="0:30"    ["large test WNLI"]="0:30"    ["small train WNLI"]="0:30"    ["large train WNLI"]="0:30"  )


for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MIMIC-a' 'MIMIC-d' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'inp-grad-abs' 'inp-grad-sign' 'int-grad-abs' 'int-grad-sign'
        do
            for masking_strategy in 'half-det' 'uni'
            do
                for split in 'test' 'train'
                do
                    submit_seeds "${time[${size[$model]} $split $dataset]}" "$seeds" $(job_script gpu) \
                        experiments/ood.py \
                        --model "${model}" \
                        --dataset "${dataset}" \
                        --max-masking-ratio 100 \
                        --masking-strategy "${masking_strategy}" \
                        --explainer "${explainer}" \
                        --split "${split}" \
                        --ood 'MaSF' \
                        --jit-compile \
                        --save-annotated-datasets
                done
            done
        done
    done
done
