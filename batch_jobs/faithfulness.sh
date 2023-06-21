#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )


declare -A algo=( ["rand"]="rand"
                  ["grad-l1"]="grad" ["grad-l2"]="grad"
                  ["inp-grad-sign"]="inp-grad" ["inp-grad-abs"]="inp-grad"
                  ["int-grad-sign"]="int-grad" ["int-grad-abs"]="int-grad"
                  ["loo-sign"]="loo" ["loo-abs"]="loo"
                  ["beam-sign-10"]="beam-10" ["beam-abs"]="beam-10"
                  ["beam-sign-20"]="beam-20" ["beam-abs"]="beam-20"
                  ["beam-sign-50"]="beam-50" ["beam-abs"]="beam-50" )

#                   V100                                 V100
declare -A time=( # ["small test rand bAbI-1"]="0:02"  ["large test rand bAbI-1"]="0:03"
                    ["small test rand bAbI-1"]="0:10"  ["large test rand bAbI-1"]="0:10"
                  # ["small test rand bAbI-2"]="0:03"  ["large test rand bAbI-2"]="0:04"
                    ["small test rand bAbI-2"]="0:10"  ["large test rand bAbI-2"]="0:10"
                  # ["small test rand bAbI-3"]="0:04"  ["large test rand bAbI-3"]="0:04"
                    ["small test rand bAbI-3"]="0:10"  ["large test rand bAbI-3"]="0:10"
                  # ["small test rand BoolQ"]="0:09"   ["large test rand BoolQ"]="0:09"
                    ["small test rand BoolQ"]="0:15"   ["large test rand BoolQ"]="0:20"
                  # ["small test rand CB"]="0:05"      ["large test rand CB"]="0:06"
                    ["small test rand CB"]="0:15"      ["large test rand CB"]="0:15"
                  # ["small test rand CoLA"]="0:05"    ["large test rand CoLA"]="0:07"
                    ["small test rand CoLA"]="0:20"    ["large test rand CoLA"]="0:20"
                  # ["small test rand IMDB"]="0:18"    ["large test rand IMDB"]="0:41"
                    ["small test rand IMDB"]="0:30"    ["large test rand IMDB"]="1:00"
                  # ["small test rand MIMIC-a"]="0:03" ["large test rand MIMIC-a"]="0:04"
                    ["small test rand MIMIC-a"]="0:10" ["large test rand MIMIC-a"]="0:10"
                  # ["small test rand MIMIC-d"]="0:03" ["large test rand MIMIC-d"]="0:05"
                    ["small test rand MIMIC-d"]="0:10" ["large test rand MIMIC-d"]="0:10"
                  # ["small test rand MNLI"]="0:07"    ["large test rand MNLI"]="0:50"
                    ["small test rand MNLI"]="0:15"    ["large test rand MNLI"]="1:00"
                  # ["small test rand MRPC"]="0:06"    ["large test rand MRPC"]="0:07"
                    ["small test rand MRPC"]="0:15"    ["large test rand MRPC"]="0:20"
                  # ["small test rand QNLI"]="0:07"    ["large test rand QNLI"]="0:08"
                    ["small test rand QNLI"]="0:15"    ["large test rand QNLI"]="0:20"
                  # ["small test rand QQP"]="0:05"     ["large test rand QQP"]="0:14"
                    ["small test rand QQP"]="0:15"     ["large test rand QQP"]="0:30"
                  # ["small test rand RTE"]="0:07"     ["large test rand RTE"]="0:05"
                    ["small test rand RTE"]="0:20"     ["large test rand RTE"]="0:20"
                  # ["small test rand SST2"]="0:05"    ["large test rand SST2"]="0:06"
                    ["small test rand SST2"]="0:20"    ["large test rand SST2"]="0:20"
                  # ["small test rand SNLI"]="0:07"    ["large test rand SNLI"]="0:09"
                    ["small test rand SNLI"]="0:15"    ["large test rand SNLI"]="1:00"
                  # ["small test rand WNLI"]="0:05"    ["large test rand WNLI"]="0:06"
                    ["small test rand WNLI"]="0:20"    ["large test rand WNLI"]="0:20"

                  # ["small test grad bAbI-1"]="0:04"  ["large test grad bAbI-1"]="0:07""
                    ["small test grad bAbI-1"]="0:15"  ["large test grad bAbI-1"]="0:15"
                  # ["small test grad bAbI-2"]="0:06"  ["large test grad bAbI-2"]="0:09"
                    ["small test grad bAbI-2"]="0:15"  ["large test grad bAbI-2"]="0:15"
                  # ["small test grad bAbI-3"]="0:05"  ["large test grad bAbI-3"]="0:08"
                    ["small test grad bAbI-3"]="0:15"  ["large test grad bAbI-3"]="0:15"
                  # ["small test grad BoolQ"]="0:12"   ["large test grad BoolQ"]="0:20"
                    ["small test grad BoolQ"]="0:20"   ["large test grad BoolQ"]="0:30"
                  # ["small test grad CB"]="0:08"      ["large test grad CB"]="0:08"
                    ["small test grad CB"]="0:15"      ["large test grad CB"]="0:15"
                  # ["small test grad CoLA"]="0:20"    ["large test grad CoLA"]="0:10"
                    ["small test grad CoLA"]="0:30"    ["large test grad CoLA"]="0:20"
                  # ["small test grad IMDB"]="0:55"    ["large test grad IMDB"]="2:01"
                    ["small test grad IMDB"]="1:10"    ["large test grad IMDB"]="2:30"
                  # ["small test grad MIMIC-a"]="0:05" ["large test grad MIMIC-a"]="0:10"
                    ["small test grad MIMIC-a"]="0:20" ["large test grad MIMIC-a"]="0:20"
                  # ["small test grad MIMIC-d"]="0:06" ["large test grad MIMIC-d"]="0:12"
                    ["small test grad MIMIC-d"]="0:20" ["large test grad MIMIC-d"]="0:25"
                  # ["small test grad MNLI"]="0:12"    ["large test grad MNLI"]="0:30"
                    ["small test grad MNLI"]="0:25"    ["large test grad MNLI"]="0:40"
                  # ["small test grad MRPC"]="0:09"    ["large test grad MRPC"]="0:10"
                    ["small test grad MRPC"]="0:20"    ["large test grad MRPC"]="0:20"
                  # ["small test grad QNLI"]="0:10"    ["large test grad QNLI"]="0:15"
                    ["small test grad QNLI"]="0:20"    ["large test grad QNLI"]="0:30"
                  # ["small test grad QQP"]="0:20"     ["large test grad QQP"]="0:40"
                    ["small test grad QQP"]="0:40"     ["large test grad QQP"]="1:00"
                  # ["small test grad RTE"]="0:06"     ["large test grad RTE"]="0:10"
                    ["small test grad RTE"]="0:20"     ["large test grad RTE"]="0:20"
                  # ["small test grad SST2"]="0:07"    ["large test grad SST2"]="0:09"
                    ["small test grad SST2"]="0:15"    ["large test grad SST2"]="0:20"
                  # ["small test grad SNLI"]="0:12"    ["large test grad SNLI"]="0:17"
                    ["small test grad SNLI"]="0:25"    ["large test grad SNLI"]="0:40"
                  # ["small test grad WNLI"]="0:07"    ["large test grad WNLI"]="0:08"
                    ["small test grad WNLI"]="0:20"    ["large test grad WNLI"]="0:20"

                  # ["small test inp-grad bAbI-1"]="0:04"  ["large test inp-grad bAbI-1"]="0:07""
                    ["small test inp-grad bAbI-1"]="0:15"  ["large test inp-grad bAbI-1"]="0:20"
                  # ["small test inp-grad bAbI-2"]="0:05"  ["large test inp-grad bAbI-2"]="0:09"
                    ["small test inp-grad bAbI-2"]="0:15"  ["large test inp-grad bAbI-2"]="0:20"
                  # ["small test inp-grad bAbI-3"]="0:04"  ["large test inp-grad bAbI-3"]="0:08"
                    ["small test inp-grad bAbI-3"]="0:15"  ["large test inp-grad bAbI-3"]="0:20"
                  # ["small test inp-grad BoolQ"]="0:11"   ["large test inp-grad BoolQ"]="0:18"
                    ["small test inp-grad BoolQ"]="0:15"   ["large test inp-grad BoolQ"]="0:30"
                  # ["small test inp-grad CB"]="0:06"      ["large test inp-grad CB"]="0:07"
                    ["small test inp-grad CB"]="0:10"      ["large test inp-grad CB"]="0:15"
                  # ["small test inp-grad CoLA"]="0:07"    ["large test inp-grad CoLA"]="0:10"
                    ["small test inp-grad CoLA"]="0:15"    ["large test inp-grad CoLA"]="0:30"
                  # ["small test inp-grad IMDB"]="0:42"    ["large test inp-grad IMDB"]="1:42"
                    ["small test inp-grad IMDB"]="1:00"    ["large test inp-grad IMDB"]="2:00"
                  # ["small test inp-grad MIMIC-a"]="0:04" ["large test inp-grad MIMIC-a"]="0:09"
                    ["small test inp-grad MIMIC-a"]="0:20" ["large test inp-grad MIMIC-a"]="0:20"
                  # ["small test inp-grad MIMIC-d"]="0:05" ["large test inp-grad MIMIC-d"]="0:11"
                    ["small test inp-grad MIMIC-d"]="0:20" ["large test inp-grad MIMIC-d"]="0:30"
                  # ["small test inp-grad MNLI"]="0:11"    ["large test inp-grad MNLI"]="0:18"
                    ["small test inp-grad MNLI"]="0:15"    ["large test inp-grad MNLI"]="0:30"
                  # ["small test inp-grad MRPC"]="0:08"    ["large test inp-grad MRPC"]="0:09"
                    ["small test inp-grad MRPC"]="0:20"    ["large test inp-grad MRPC"]="0:20"
                  # ["small test inp-grad QNLI"]="0:10"    ["large test inp-grad QNLI"]="0:16"
                    ["small test inp-grad QNLI"]="0:20"    ["large test inp-grad QNLI"]="0:30"
                  # ["small test inp-grad QQP"]="0:15"     ["large test inp-grad QQP"]="0:39"
                    ["small test inp-grad QQP"]="0:30"     ["large test inp-grad QQP"]="1:00"
                  # ["small test inp-grad RTE"]="0:07"     ["large test inp-grad RTE"]="0:10"
                    ["small test inp-grad RTE"]="0:20"     ["large test inp-grad RTE"]="0:30"
                  # ["small test inp-grad SST2"]="0:08"    ["large test inp-grad SST2"]="0:09"
                    ["small test inp-grad SST2"]="0:20"    ["large test inp-grad SST2"]="0:20"
                  # ["small test inp-grad SNLI"]="0:11"    ["large test inp-grad SNLI"]="0:15"
                    ["small test inp-grad SNLI"]="0:15"    ["large test inp-grad SNLI"]="0:30"
                  # ["small test inp-grad WNLI"]="0:06"    ["large test inp-grad WNLI"]="0:08"
                    ["small test inp-grad WNLI"]="0:20"    ["large test inp-grad WNLI"]="0:20"

                  # ["small test int-grad bAbI-1"]="0:07"  ["large test int-grad bAbI-1"]="0:14""
                    ["small test int-grad bAbI-1"]="0:20"  ["large test int-grad bAbI-1"]="0:25"
                  # ["small test int-grad bAbI-2"]="0:14"  ["large test int-grad bAbI-2"]="0:31"
                    ["small test int-grad bAbI-2"]="0:30"  ["large test int-grad bAbI-2"]="0:50"
                  # ["small test int-grad bAbI-3"]="0:21"  ["large test int-grad bAbI-3"]="0:57"
                    ["small test int-grad bAbI-3"]="0:40"  ["large test int-grad bAbI-3"]="1:10"
                  # ["small test int-grad BoolQ"]="0:48"   ["large test int-grad BoolQ"]="1:56"
                    ["small test int-grad BoolQ"]="1:00"   ["large test int-grad BoolQ"]="2:30"
                  # ["small test int-grad CB"]="0:06"      ["large test int-grad CB"]="0:09"
                    ["small test int-grad CB"]="0:15"      ["large test int-grad CB"]="0:20"
                  # ["small test int-grad CoLA"]="0:09"    ["large test int-grad CoLA"]="0:14"
                    ["small test int-grad CoLA"]="0:20"    ["large test int-grad CoLA"]="0:30"
                  # ["small test int-grad IMDB"]="7:39"    ["large test int-grad IMDB"]="22:07"
                    ["small test int-grad IMDB"]="8:30"    ["large test int-grad IMDB"]="23:30"
                  # ["small test int-grad MIMIC-a"]="0:26" ["large test int-grad MIMIC-a"]="1:11"
                    ["small test int-grad MIMIC-a"]="0:40" ["large test int-grad MIMIC-a"]="1:30"
                  # ["small test int-grad MIMIC-d"]="0:34" ["large test int-grad MIMIC-d"]="1:37"
                    ["small test int-grad MIMIC-d"]="1:00" ["large test int-grad MIMIC-d"]="2:00"
                  # ["small test int-grad MNLI"]="0:43"    ["large test int-grad MNLI"]="1:47"
                    ["small test int-grad MNLI"]="1:00"    ["large test int-grad MNLI"]="2:30"
                  # ["small test int-grad MRPC"]="0:08"    ["large test int-grad MRPC"]="0:14"
                    ["small test int-grad MRPC"]="0:20"    ["large test int-grad MRPC"]="0:30"
                  # ["small test int-grad QNLI"]="0:29"    ["large test int-grad QNLI"]="1:14"
                    ["small test int-grad QNLI"]="0:50"    ["large test int-grad QNLI"]="1:40"
                  # ["small test int-grad QQP"]="2:10"     ["large test int-grad QQP"]="5:40"
                    ["small test int-grad QQP"]="3:30"     ["large test int-grad QQP"]="6:40"
                  # ["small test int-grad RTE"]="0:09"     ["large test int-grad RTE"]="0:15"
                    ["small test int-grad RTE"]="0:20"     ["large test int-grad RTE"]="1:30"
                  # ["small test int-grad SST2"]="0:08"    ["large test int-grad SST2"]="0:14"
                    ["small test int-grad SST2"]="0:20"    ["large test int-grad SST2"]="0:30"
                  # ["small test int-grad SNLI"]="0:30"    ["large test int-grad SNLI"]="1:06"
                    ["small test int-grad SNLI"]="1:00"    ["large test int-grad SNLI"]="2:30"
                  # ["small test int-grad WNLI"]="0:07"    ["large test int-grad WNLI"]="0:09"
                    ["small test int-grad WNLI"]="0:20"    ["large test int-grad WNLI"]="0:20"

                  # ["small test loo bAbI-1"]="0:29"  ["large test loo bAbI-1"]="0:54"
                    ["small test loo bAbI-1"]="0:50"  ["large test loo bAbI-1"]="1:20"
                  # ["small test loo bAbI-2"]="0:44"  ["large test loo bAbI-2"]="1:32"
                    ["small test loo bAbI-2"]="1:10"  ["large test loo bAbI-2"]="1:50"
                  # ["small test loo bAbI-3"]="1:14"  ["large test loo bAbI-3"]="3:30"
                    ["small test loo bAbI-3"]="1:40"  ["large test loo bAbI-3"]="4:20"
                  # ["small test loo BoolQ"]="0:48"   ["large test loo BoolQ"]="0:47"
                    ["small test loo BoolQ"]="1:10"   ["large test loo BoolQ"]="1:10"
                  # ["small test loo CB"]="0:14"      ["large test loo CB"]="0:52"
                    ["small test loo CB"]="0:30"      ["large test loo CB"]="1:20"
                  # ["small test loo CoLA"]="0:15"    ["large test loo CoLA"]="0:24"
                    ["small test loo CoLA"]="0:30"    ["large test loo CoLA"]="0:50"
                  # ["small test loo IMDB"]="?:??"    ["large test loo IMDB"]="?:??"
                    ["small test loo IMDB"]="40:00"   ["large test loo IMDB"]="80:00"
                  # ["small test loo MIMIC-a"]="2:28" ["large test loo MIMIC-a"]="7:12"
                    ["small test loo MIMIC-a"]="3:00" ["large test loo MIMIC-a"]="8:20"
                  # ["small test loo MIMIC-d"]="3:22" ["large test loo MIMIC-d"]="10:00"
                    ["small test loo MIMIC-d"]="4:00" ["large test loo MIMIC-d"]="11:30"
                  # ["small test loo MNLI"]="0:27"    ["large test loo MNLI"]="0:57"
                    ["small test loo MNLI"]="1:00"    ["large test loo MNLI"]="1:30"
                  # ["small test loo MRPC"]="0:14"    ["large test loo MRPC"]="0:23"
                    ["small test loo MRPC"]="0:30"    ["large test loo MRPC"]="0:50"
                  # ["small test loo QNLI"]="0:25"    ["large test loo QNLI"]="0:52"
                    ["small test loo QNLI"]="0:50"    ["large test loo QNLI"]="1:30"
                  # ["small test loo QQP"]="0:45"     ["large test loo QQP"]="1:40"
                    ["small test loo QQP"]="1:20"     ["large test loo QQP"]="2:40"
                  # ["small test loo RTE"]="0:15"     ["large test loo RTE"]="0:28"
                    ["small test loo RTE"]="1:20"     ["large test loo RTE"]="0:50"
                  # ["small test loo SST2"]="0:15"    ["large test loo SST2"]="0:27"
                    ["small test loo SST2"]="0:30"    ["large test loo SST2"]="0:50"
                  # ["small test loo SNLI"]="0:18"    ["large test loo SNLI"]="0:36"
                    ["small test loo SNLI"]="0:50"    ["large test loo SNLI"]="1:20"
                  # ["small test loo WNLI"]="?:??"    ["large test loo WNLI"]="?:??"
                    ["small test loo WNLI"]="1:20"    ["large test loo WNLI"]="1:20"

                  # ["small test beam-10 bAbI-1"]="1:00"    ["large test beam-10 bAbI-1"]="2:29"
                    ["small test beam-10 bAbI-1"]="1:30"    ["large test beam-10 bAbI-1"]="3:00"
                  # ["small test beam-10 bAbI-2"]="30:??"   ["large test beam-10 bAbI-2"]="62:??"
                    ["small test beam-10 bAbI-2"]="40:00"   ["large test beam-10 bAbI-2"]="70:00"
                  # ["small test beam-10 bAbI-3"]="451:??"  ["large test beam-10 bAbI-3"]="UNK:??"
                    ["small test beam-10 bAbI-3"]="?:??"    ["large test beam-10 bAbI-3"]="?:??"
                  # ["small test beam-10 BoolQ"]="0:40"     ["large test beam-10 BoolQ"]="1:28"
                    ["small test beam-10 BoolQ"]="1:00"     ["large test beam-10 BoolQ"]="2:00"
                  # ["small test beam-10 CB"]="0:53"        ["large test beam-10 CB"]="2:12"
                    ["small test beam-10 CB"]="1:30"        ["large test beam-10 CB"]="3:00"
                  # ["small test beam-10 CoLA"]="0:16"      ["large test beam-10 CoLA"]="0:25"
                    ["small test beam-10 CoLA"]="1:00"      ["large test beam-10 CoLA"]="1:00"
                  # ["small test beam-10 IMDB"]="?:??"      ["large test beam-10 IMDB"]="?:??"
                    ["small test beam-10 IMDB"]="?:??"      ["large test beam-10 IMDB"]="?:??"
                  # ["small test beam-10 MIMIC-a"]="UNK:??" ["large test beam-10 MIMIC-a"]="UNK:??"
                    ["small test beam-10 MIMIC-a"]="?:??"   ["large test beam-10 MIMIC-a"]="?:??"
                  # ["small test beam-10 MIMIC-d"]="UNK:??" ["large test beam-10 MIMIC-d"]="UNK:??"
                    ["small test beam-10 MIMIC-d"]="?:??"   ["large test beam-10 MIMIC-d"]="?:??"
                  # ["small test beam-10 MNLI"]="5:57"      ["large test beam-10 MNLI"]="?:??"
                    ["small test beam-10 MNLI"]="7:00"      ["large test beam-10 MNLI"]="20:00"
                  # ["small test beam-10 MRPC"]="0:19"      ["large test beam-10 MRPC"]="0:38"
                    ["small test beam-10 MRPC"]="1:00"      ["large test beam-10 MRPC"]="1:00"
                  # ["small test beam-10 QNLI"]="6:50"      ["large test beam-10 QNLI"]="16:00"
                    ["small test beam-10 QNLI"]="8:00"      ["large test beam-10 QNLI"]="20:00"
                  # ["small test beam-10 QQP"]="4:50"       ["large test beam-10 QQP"]="12:00"
                    ["small test beam-10 QQP"]="6:00"       ["large test beam-10 QQP"]="15:00"
                  # ["small test beam-10 RTE"]="1:38"       ["large test beam-10 RTE"]="5:??"
                    ["small test beam-10 RTE"]="2:00"       ["large test beam-10 RTE"]="6:00"
                  # ["small test beam-10 SST2"]="0:22"      ["large test beam-10 SST2"]="0:49"
                    ["small test beam-10 SST2"]="1:00"      ["large test beam-10 SST2"]="1:30"
                  # ["small test beam-10 SNLI"]="1:15"      ["large test beam-10 SNLI"]="2:48"
                    ["small test beam-10 SNLI"]="2:00"      ["large test beam-10 SNLI"]="3:30"
                  # ["small test beam-10 WNLI"]="?:??"      ["large test beam-10 WNLI"]="?:??"
                    ["small test beam-10 WNLI"]="?:??"      ["large test beam-10 WNLI"]="?:??" )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'MIMIC-a' 'MIMIC-d' 'MRPC' 'RTE' 'SST2' 'SNLI' 'IMDB' 'MNLI' 'QNLI' 'QQP' # 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'inp-grad-abs' 'inp-grad-sign' 'int-grad-abs' 'int-grad-sign' 'loo-sign' 'loo-abs' 'beam-sign-10'
        do
            for split in 'test'
            do
                for max_masking_ratio in 100 0
                do
                    for seeds in 0 1 2 3 4
                    do
                      submit_seeds "${time[${size[$model]} $split ${algo[$explainer]} $dataset]}" "$seeds" $(job_script gpu) \
                          experiments/faithfulness.py \
                          --model "${model}" \
                          --dataset "${dataset}" \
                          --max-masking-ratio "${max_masking_ratio}" \
                          --masking-strategy 'half-det' \
                          --validation-dataset 'both' \
                          --explainer "${explainer}" \
                          --split "${split}" \
                          --jit-compile \
                          --save-masked-datasets
                    done
                done
            done
        done
    done
done
