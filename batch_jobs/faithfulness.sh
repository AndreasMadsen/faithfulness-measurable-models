#!/bin/bash
# jobs: 11
source "batch_jobs/_job_script.sh"
seeds="0 1 2 3 4"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )


declare -A algo=( ["rand"]="rand"
                  ["grad-l1"]="grad" ["grad-l2"]="grad"
                  ["inp-grad-sign"]="inp-grad" ["inp-grad-abs"]="inp-grad"
                  ["int-grad-sign"]="int-grad" ["int-grad-abs"]="int-grad" )

#                   V100                             V100
declare -A time=( # ["small rand bAbI-1"]="0:02:0"  ["large rand bAbI-1"]="0:03:0"
                    ["small rand bAbI-1"]="0:10:0"  ["large rand bAbI-1"]="0:10:0"
                  # ["small rand bAbI-2"]="0:03:0"  ["large rand bAbI-2"]="0:04:0"
                    ["small rand bAbI-2"]="0:10:0"  ["large rand bAbI-2"]="0:10:0"
                  # ["small rand bAbI-3"]="0:04:0"  ["large rand bAbI-3"]="0:04:0"
                    ["small rand bAbI-3"]="0:10:0"  ["large rand bAbI-3"]="0:10:0"
                  # ["small rand BoolQ"]="0:04:0"   ["large rand BoolQ"]="0:06:0"
                    ["small rand BoolQ"]="0:10:0"   ["large rand BoolQ"]="0:15:0"
                  # ["small rand CB"]="0:02:0"      ["large rand CB"]="0:03:0"
                    ["small rand CB"]="0:10:0"      ["large rand CB"]="0:10:0"
                  # ["small rand CoLA"]="0:03:0"    ["large rand CoLA"]="0:03:0"
                    ["small rand CoLA"]="0:10:0"    ["large rand CoLA"]="0:10:0"
                  # ["small rand IMDB"]="0:14:0"    ["large rand IMDB"]="0:??:0"
                    ["small rand IMDB"]="0:30:0"    ["large rand IMDB"]="1:30:0"
                  # ["small rand MIMIC-a"]="0:03:0" ["large rand MIMIC-a"]="0:04:0"
                    ["small rand MIMIC-a"]="0:10:0" ["large rand MIMIC-a"]="0:10:0"
                  # ["small rand MIMIC-d"]="0:03:0" ["large rand MIMIC-d"]="0:05:0"
                    ["small rand MIMIC-d"]="0:10:0" ["large rand MIMIC-d"]="0:10:0"
                  # ["small rand MNLI"]="0:04:0"    ["large rand MNLI"]="0:06:0"
                    ["small rand MNLI"]="0:10:0"    ["large rand MNLI"]="0:15:0"
                  # ["small rand MRPC"]="0:03:0"    ["large rand MRPC"]="0:04:0"
                    ["small rand MRPC"]="0:10:0"    ["large rand MRPC"]="0:10:0"
                  # ["small rand QNLI"]="0:03:0"    ["large rand QNLI"]="0:??:0"
                    ["small rand QNLI"]="0:10:0"    ["large rand QNLI"]="0:30:0"
                  # ["small rand QQP"]="0:05:0"     ["large rand QQP"]="0:14:0"
                    ["small rand QQP"]="0:15:0"     ["large rand QQP"]="0:30:0"
                  # ["small rand RTE"]="0:03:0"     ["large rand RTE"]="0:??:0"
                    ["small rand RTE"]="0:10:0"     ["large rand RTE"]="0:30:0"
                  # ["small rand SST2"]="0:03:0"    ["large rand SST2"]="0:03:0"
                    ["small rand SST2"]="0:10:0"    ["large rand SST2"]="0:10:0"
                  # ["small rand WNLI"]="0:02:0"    ["large rand WNLI"]="0:03:0"
                    ["small rand WNLI"]="0:10:0"    ["large rand WNLI"]="0:10:0"

                  # ["small grad bAbI-1"]="0:04:0"  ["large grad bAbI-1"]="0:07:0"
                    ["small grad bAbI-1"]="0:15:0"  ["large grad bAbI-1"]="0:15:0"
                  # ["small grad bAbI-2"]="0:06:0"  ["large grad bAbI-2"]="0:09:0"
                    ["small grad bAbI-2"]="0:15:0"  ["large grad bAbI-2"]="0:15:0"
                  # ["small grad bAbI-3"]="0:05:0"  ["large grad bAbI-3"]="0:08:0"
                    ["small grad bAbI-3"]="0:15:0"  ["large grad bAbI-3"]="0:15:0"
                  # ["small grad BoolQ"]="0:09:0"   ["large grad BoolQ"]="0:17:0"
                    ["small grad BoolQ"]="0:20:0"   ["large grad BoolQ"]="0:30:0"
                  # ["small grad CB"]="0:04:0"      ["large grad CB"]="0:06:0"
                    ["small grad CB"]="0:15:0"      ["large grad CB"]="0:15:0"
                  # ["small grad CoLA"]="0:05:0"    ["large grad CoLA"]="0:08:0"
                    ["small grad CoLA"]="0:15:0"    ["large grad CoLA"]="0:20:0"
                  # ["small grad IMDB"]="0:30:0"    ["large grad IMDB"]="0:40:0"
                    ["small grad IMDB"]="0:50:0"    ["large grad IMDB"]="1:00:0"
                  # ["small grad MIMIC-a"]="0:05:0" ["large grad MIMIC-a"]="0:10:0"
                    ["small grad MIMIC-a"]="0:20:0" ["large grad MIMIC-a"]="0:20:0"
                  # ["small grad MIMIC-d"]="0:06:0" ["large grad MIMIC-d"]="0:12:0"
                    ["small grad MIMIC-d"]="0:20:0" ["large grad MIMIC-d"]="0:25:0"
                  # ["small grad MNLI"]="0:09:0"    ["large grad MNLI"]="0:17:0"
                    ["small grad MNLI"]="0:20:0"    ["large grad MNLI"]="0:30:0"
                  # ["small grad MRPC"]="0:05:0"    ["large grad MRPC"]="0:07:0"
                    ["small grad MRPC"]="0:15:0"    ["large grad MRPC"]="0:15:0"
                  # ["small grad QNLI"]="0:07:0"    ["large grad QNLI"]="0:??:0"
                    ["small grad QNLI"]="0:15:0"    ["large grad QNLI"]="0:30:0"
                  # ["small grad QQP"]="0:12:0"     ["large grad QQP"]="0:??:0"
                    ["small grad QQP"]="0:20:0"     ["large grad QQP"]="0:30:0"
                  # ["small grad RTE"]="0:05:0"     ["large grad RTE"]="0:??:0"
                    ["small grad RTE"]="0:15:0"     ["large grad RTE"]="0:30:0"
                  # ["small grad SST2"]="0:05:0"    ["large grad SST2"]="0:08:0"
                    ["small grad SST2"]="0:15:0"    ["large grad SST2"]="0:20:0"
                  # ["small grad WNLI"]="0:04:0"    ["large grad WNLI"]="0:07:0"
                    ["small grad WNLI"]="0:10:0"    ["large grad WNLI"]="0:15:0"

                  # ["small inp-grad bAbI-1"]="0:04:0"  ["large inp-grad bAbI-1"]="0:07:0"
                    ["small inp-grad bAbI-1"]="0:15:0"  ["large inp-grad bAbI-1"]="0:20:0"
                  # ["small inp-grad bAbI-2"]="0:05:0"  ["large inp-grad bAbI-2"]="0:09:0"
                    ["small inp-grad bAbI-2"]="0:15:0"  ["large inp-grad bAbI-2"]="0:20:0"
                  # ["small inp-grad bAbI-3"]="0:04:0"  ["large inp-grad bAbI-3"]="0:08:0"
                    ["small inp-grad bAbI-3"]="0:15:0"  ["large inp-grad bAbI-3"]="0:20:0"
                  # ["small inp-grad BoolQ"]="0:08:0"   ["large inp-grad BoolQ"]="0:16:0"
                    ["small inp-grad BoolQ"]="0:15:0"   ["large inp-grad BoolQ"]="0:30:0"
                  # ["small inp-grad CB"]="0:03:0"      ["large inp-grad CB"]="0:05:0"
                    ["small inp-grad CB"]="0:10:0"      ["large inp-grad CB"]="0:10:0"
                  # ["small inp-grad CoLA"]="0:05:0"    ["large inp-grad CoLA"]="0:08:0"
                    ["small inp-grad CoLA"]="0:15:0"    ["large inp-grad CoLA"]="0:30:0"
                  # ["small inp-grad IMDB"]="0:41:0"    ["large inp-grad IMDB"]="1:05:0" # approx
                    ["small inp-grad IMDB"]="1:00:0"    ["large inp-grad IMDB"]="1:30:0"
                  # ["small inp-grad MIMIC-a"]="0:04:0" ["large inp-grad MIMIC-a"]="0:09:0"
                    ["small inp-grad MIMIC-a"]="0:20:0" ["large inp-grad MIMIC-a"]="0:20:0"
                  # ["small inp-grad MIMIC-d"]="0:05:0" ["large inp-grad MIMIC-d"]="0:11:0"
                    ["small inp-grad MIMIC-d"]="0:20:0" ["large inp-grad MIMIC-d"]="0:30:0"
                  # ["small inp-grad MNLI"]="0:07:0"    ["large inp-grad MNLI"]="0:16:0"
                    ["small inp-grad MNLI"]="0:15:0"    ["large inp-grad MNLI"]="0:30:0"
                  # ["small inp-grad MRPC"]="0:04:0"    ["large inp-grad MRPC"]="0:07:0"
                    ["small inp-grad MRPC"]="0:10:0"    ["large inp-grad MRPC"]="0:15:0"
                  # ["small inp-grad QNLI"]="0:07:0"    ["large inp-grad QNLI"]="0:??:0"
                    ["small inp-grad QNLI"]="0:10:0"    ["large inp-grad QNLI"]="0:30:0"
                  # ["small inp-grad QQP"]="0:10:0"     ["large inp-grad QQP"]="0:39:0"
                    ["small inp-grad QQP"]="0:20:0"     ["large inp-grad QQP"]="1:30:0"
                  # ["small inp-grad RTE"]="0:05:0"     ["large inp-grad RTE"]="0:??:0"
                    ["small inp-grad RTE"]="0:15:0"     ["large inp-grad RTE"]="0:30:0"
                  # ["small inp-grad SST2"]="0:04:0"    ["large inp-grad SST2"]="0:06:0"
                    ["small inp-grad SST2"]="0:10:0"    ["large inp-grad SST2"]="0:15:0"
                  # ["small inp-grad WNLI"]="0:04:0"    ["large inp-grad WNLI"]="0:06:0"
                    ["small inp-grad WNLI"]="0:10:0"    ["large inp-grad WNLI"]="0:15:0"

                  # ["small int-grad bAbI-1"]="0:07:0"  ["large int-grad bAbI-1"]="0:14:0"
                    ["small int-grad bAbI-1"]="0:20:0"  ["large int-grad bAbI-1"]="0:25:0"
                  # ["small int-grad bAbI-2"]="0:14:0"  ["large int-grad bAbI-2"]="0:31:0"
                    ["small int-grad bAbI-2"]="0:30:0"  ["large int-grad bAbI-2"]="0:50:0"
                  # ["small int-grad bAbI-3"]="0:21:0"  ["large int-grad bAbI-3"]="0:57:0"
                    ["small int-grad bAbI-3"]="0:40:0"  ["large int-grad bAbI-3"]="1:10:0"
                  # ["small int-grad BoolQ"]="0:45:0"   ["large int-grad BoolQ"]="1:40:0" # approx
                    ["small int-grad BoolQ"]="1:00:0"   ["large int-grad BoolQ"]="2:30:0"
                  # ["small int-grad CB"]="0:04:0"      ["large int-grad CB"]="0:06:0"
                    ["small int-grad CB"]="0:10:0"      ["large int-grad CB"]="0:15:0"
                  # ["small int-grad CoLA"]="0:07:0"    ["large int-grad CoLA"]="0:12:0"
                    ["small int-grad CoLA"]="0:15:0"    ["large int-grad CoLA"]="0:30:0"
                  # ["small int-grad IMDB"]="7:30:0"    ["large int-grad IMDB"]="21:30:0" # approx
                    ["small int-grad IMDB"]="8:30:0"    ["large int-grad IMDB"]="23:30:0"
                  # ["small int-grad MIMIC-a"]="0:26:0" ["large int-grad MIMIC-a"]="1:11:0"
                    ["small int-grad MIMIC-a"]="0:40:0" ["large int-grad MIMIC-a"]="1:30:0"
                  # ["small int-grad MIMIC-d"]="0:34:0" ["large int-grad MIMIC-d"]="1:37:0"
                    ["small int-grad MIMIC-d"]="1:00:0" ["large int-grad MIMIC-d"]="2:00:0"
                  # ["small int-grad MNLI"]="0:42:0"    ["large int-grad MNLI"]="2:00:0" # approx
                    ["small int-grad MNLI"]="1:00:0"    ["large int-grad MNLI"]="2:30:0"
                  # ["small int-grad MRPC"]="0:06:0"    ["large int-grad MRPC"]="0:11:0"
                    ["small int-grad MRPC"]="0:15:0"    ["large int-grad MRPC"]="0:20:0"
                  # ["small int-grad QNLI"]="0:28:0"    ["large int-grad QNLI"]="0:??:0"
                    ["small int-grad QNLI"]="0:50:0"    ["large int-grad QNLI"]="2:30:0"
                  # ["small int-grad QQP"]="2:10:0"     ["large int-grad QQP"]="5:40:0" # approx
                    ["small int-grad QQP"]="3:30:0"     ["large int-grad QQP"]="6:40:0"
                  # ["small int-grad RTE"]="0:06:0"     ["large int-grad RTE"]="0:??:0"
                    ["small int-grad RTE"]="0:15:0"     ["large int-grad RTE"]="1:30:0"
                  # ["small int-grad SST2"]="0:06:0"    ["large int-grad SST2"]="0:10:0"
                    ["small int-grad SST2"]="0:15:0"    ["large int-grad SST2"]="0:20:0"
                  # ["small int-grad WNLI"]="0:04:0"    ["large int-grad WNLI"]="0:06:0"
                    ["small int-grad WNLI"]="0:10:0"    ["large int-grad WNLI"]="0:15:0" )


for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        for explainer in 'rand' 'grad-l1' 'grad-l2' 'inp-grad-abs' 'inp-grad-sign' 'int-grad-abs' 'int-grad-sign'
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
