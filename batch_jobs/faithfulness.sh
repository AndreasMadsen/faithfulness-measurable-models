#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )


declare -A algo=( ["rand"]="rand"
                  ["grad-l1"]="grad" ["grad-l2"]="grad"
                  ["x-grad-sign"]="x-grad" ["x-grad-abs"]="x-grad"
                  ["int-grad-sign"]="int-grad" ["int-grad-abs"]="ing-grad" )

#                   V100                           V100
declare -A time=( # ["small rand BoolQ"]="0:04:0" ["large rand BoolQ"]="0:06:0"
                    ["small rand BoolQ"]="0:10:0" ["large rand BoolQ"]="0:15:0"
                  # ["small rand CB"]="0:02:0"    ["large rand CB"]="0:03:0"
                    ["small rand CB"]="0:10:0"    ["large rand CB"]="0:10:0"
                  # ["small rand CoLA"]="0:03:0"  ["large rand CoLA"]="0:03:0"
                    ["small rand CoLA"]="0:10:0"  ["large rand CoLA"]="0:10:0"
                  # ["small rand IMDB"]="0:14:0"  ["large rand IMDB"]="0:??:0"
                    ["small rand IMDB"]="0:30:0"  ["large rand IMDB"]="1:30:0"
                  # ["small rand MNLI"]="0:04:0"  ["large rand MNLI"]="0:06:0"
                    ["small rand MNLI"]="0:10:0"  ["large rand MNLI"]="0:15:0"
                  # ["small rand MRPC"]="0:03:0"  ["large rand MRPC"]="0:04:0"
                    ["small rand MRPC"]="0:10:0"  ["large rand MRPC"]="0:10:0"
                  # ["small rand QNLI"]="0:03:0"  ["large rand QNLI"]="0:??:0"
                    ["small rand QNLI"]="0:10:0"  ["large rand QNLI"]="0:30:0"
                  # ["small rand QQP"]="0:05:0"   ["large rand QQP"]="0:14:0"
                    ["small rand QQP"]="0:15:0"   ["large rand QQP"]="0:30:0"
                  # ["small rand RTE"]="0:03:0"   ["large rand RTE"]="0:??:0"
                    ["small rand RTE"]="0:10:0"   ["large rand RTE"]="0:30:0"
                  # ["small rand SST2"]="0:03:0"  ["large rand SST2"]="0:03:0"
                    ["small rand SST2"]="0:10:0"  ["large rand SST2"]="0:10:0"
                  # ["small rand WNLI"]="0:02:0"  ["large rand WNLI"]="0:03:0"
                    ["small rand WNLI"]="0:10:0"  ["large rand WNLI"]="0:10:0"

                  # ["small grad BoolQ"]="0:09:0" ["large grad BoolQ"]="0:17:0"
                    ["small grad BoolQ"]="0:20:0" ["large grad BoolQ"]="0:30:0"
                  # ["small grad CB"]="0:04:0"    ["large grad CB"]="0:06:0"
                    ["small grad CB"]="0:15:0"    ["large grad CB"]="0:15:0"
                  # ["small grad CoLA"]="0:05:0"  ["large grad CoLA"]="0:08:0"
                    ["small grad CoLA"]="0:15:0"  ["large grad CoLA"]="0:20:0"
                  # ["small grad IMDB"]="0:??:0"  ["large grad IMDB"]="0:40:0"
                    ["small grad IMDB"]="1:30:0"  ["large grad IMDB"]="1:00:0"
                  # ["small grad MNLI"]="0:09:0"  ["large grad MNLI"]="0:17:0"
                    ["small grad MNLI"]="0:20:0"  ["large grad MNLI"]="0:30:0"
                  # ["small grad MRPC"]="0:05:0"  ["large grad MRPC"]="0:07:0"
                    ["small grad MRPC"]="0:15:0"  ["large grad MRPC"]="0:15:0"
                  # ["small grad QNLI"]="0:07:0"  ["large grad QNLI"]="0:??:0"
                    ["small grad QNLI"]="0:15:0"  ["large grad QNLI"]="0:30:0"
                  # ["small grad QQP"]="0:12:0"   ["large grad QQP"]="0:??:0"
                    ["small grad QQP"]="0:20:0"   ["large grad QQP"]="0:30:0"
                  # ["small grad RTE"]="0:05:0"   ["large grad RTE"]="0:??:0"
                    ["small grad RTE"]="0:15:0"   ["large grad RTE"]="0:30:0"
                  # ["small grad SST2"]="0:05:0"  ["large grad SST2"]="0:08:0"
                    ["small grad SST2"]="0:15:0"  ["large grad SST2"]="0:20:0"
                  # ["small grad WNLI"]="0:04:0"  ["large grad WNLI"]="0:07:0"
                    ["small grad WNLI"]="0:10:0"  ["large grad WNLI"]="0:15:0"

                  # ["small x-grad BoolQ"]="0:08:0" ["large x-grad BoolQ"]="0:16:0"
                    ["small x-grad BoolQ"]="0:15:0" ["large x-grad BoolQ"]="0:30:0"
                  # ["small x-grad CB"]="0:03:0"    ["large x-grad CB"]="0:05:0"
                    ["small x-grad CB"]="0:10:0"    ["large x-grad CB"]="0:10:0"
                  # ["small x-grad CoLA"]="0:05:0"  ["large x-grad CoLA"]="0:08:0"
                    ["small x-grad CoLA"]="0:15:0"  ["large x-grad CoLA"]="0:30:0"
                  # ["small x-grad IMDB"]="0:41:0"  ["large x-grad IMDB"]="0:??:0"
                    ["small x-grad IMDB"]="1:00:0"  ["large x-grad IMDB"]="1:30:0"
                  # ["small x-grad MNLI"]="0:07:0"  ["large x-grad MNLI"]="0:16:0"
                    ["small x-grad MNLI"]="0:15:0"  ["large x-grad MNLI"]="0:30:0"
                  # ["small x-grad MRPC"]="0:04:0"  ["large x-grad MRPC"]="0:07:0"
                    ["small x-grad MRPC"]="0:10:0"  ["large x-grad MRPC"]="0:15:0"
                  # ["small x-grad QNLI"]="0:07:0"  ["large x-grad QNLI"]="0:??:0"
                    ["small x-grad QNLI"]="0:10:0"  ["large x-grad QNLI"]="0:30:0"
                  # ["small x-grad QQP"]="0:10:0"   ["large x-grad QQP"]="0:39:0"
                    ["small x-grad QQP"]="0:20:0"   ["large x-grad QQP"]="1:30:0"
                  # ["small x-grad RTE"]="0:05:0"   ["large x-grad RTE"]="0:??:0"
                    ["small x-grad RTE"]="0:15:0"   ["large x-grad RTE"]="0:30:0"
                  # ["small x-grad SST2"]="0:04:0"  ["large x-grad SST2"]="0:06:0"
                    ["small x-grad SST2"]="0:10:0"  ["large x-grad SST2"]="0:15:0"
                  # ["small x-grad WNLI"]="0:04:0"  ["large x-grad WNLI"]="0:06:0"
                    ["small x-grad WNLI"]="0:10:0"  ["large x-grad WNLI"]="0:15:0"

                  # ["small int-grad BoolQ"]="0:45:0" ["large int-grad BoolQ"]="0:??:0"
                    ["small int-grad BoolQ"]="1:00:0" ["large int-grad BoolQ"]="2:30:0"
                  # ["small int-grad CB"]="0:04:0"    ["large int-grad CB"]="0:06:0"
                    ["small int-grad CB"]="0:10:0"    ["large int-grad CB"]="0:15:0"
                  # ["small int-grad CoLA"]="0:07:0"  ["large int-grad CoLA"]="0:12:0"
                    ["small int-grad CoLA"]="0:15:0"  ["large int-grad CoLA"]="0:30:0"
                  # ["small int-grad IMDB"]="0:??:0"  ["large int-grad IMDB"]="0:??:0"
                    ["small int-grad IMDB"]="3:30:0"  ["large int-grad IMDB"]="6:30:0"
                  # ["small int-grad MNLI"]="0:42:0"  ["large int-grad MNLI"]="0:??:0"
                    ["small int-grad MNLI"]="1:00:0"  ["large int-grad MNLI"]="2:30:0"
                  # ["small int-grad MRPC"]="0:06:0"  ["large int-grad MRPC"]="0:11:0"
                    ["small int-grad MRPC"]="0:15:0"  ["large int-grad MRPC"]="0:20:0"
                  # ["small int-grad QNLI"]="0:28:0"  ["large int-grad QNLI"]="0:??:0"
                    ["small int-grad QNLI"]="0:50:0"  ["large int-grad QNLI"]="2:30:0"
                  # ["small int-grad QQP"]="0:??:0"   ["large int-grad QQP"]="0:??:0"
                    ["small int-grad QQP"]="1:30:0"   ["large int-grad QQP"]="2:30:0"
                  # ["small int-grad RTE"]="0:06:0"   ["large int-grad RTE"]="0:??:0"
                    ["small int-grad RTE"]="0:15:0"   ["large int-grad RTE"]="1:30:0"
                  # ["small int-grad SST2"]="0:06:0"  ["large int-grad SST2"]="0:10:0"
                    ["small int-grad SST2"]="0:15:0"  ["large int-grad SST2"]="0:20:0"
                  # ["small int-grad WNLI"]="0:04:0"  ["large int-grad WNLI"]="0:06:0"
                    ["small int-grad WNLI"]="0:10:0"  ["large int-grad WNLI"]="0:15:0" )


for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'x-grad-abs' 'x-grad-sign' 'int-grad-abs' 'int-grad-sign'
        do
              submit_seeds "${time[${size[$model]} ${algo[$explainer]} $dataset]}" "$seeds" $(job_script gpu) \
                  experiments/faithfulness.py \
                  --model "${model}" \
                  --dataset "${dataset}" \
                  --max-masking-ratio 100 \
                  --masking-strategy half-det \
                  --explainer "${explainer}" \
                  --jit-compile
        done
    done
done
