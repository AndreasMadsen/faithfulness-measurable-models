source "batch_jobs/_job_script.sh"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100
declare -A time=( # ["small bAbI-1"]="0:06:0"  ["large bAbI-1"]="0:44:0"
                    ["small bAbI-1"]="0:15:0"  ["large bAbI-1"]="1:00:0"
                  # ["small bAbI-2"]="0:06:0"  ["large bAbI-2"]="1:14:0"
                    ["small bAbI-2"]="0:15:0"  ["large bAbI-2"]="1:30:0"
                  # ["small bAbI-3"]="0:06:0"  ["large bAbI-3"]="2:29:0"
                    ["small bAbI-3"]="0:15:0"  ["large bAbI-3"]="3:00:0"
                  # ["small BoolQ"]="0:05:0"   ["large BoolQ"]="1:16:0"
                    ["small BoolQ"]="1:00:0"   ["large BoolQ"]="1:50:0"
                  # ["small CB"]="0:05:0"      ["large CB"]="0:17:0"
                    ["small CB"]="0:50:0"      ["large CB"]="0:50:0"
                  # ["small CoLA"]="0:13:0"    ["large CoLA"]="2:33:0"
                    ["small CoLA"]="1:40:0"    ["large CoLA"]="3:00:0"
                  # ["small IMDB"]="0:44:0"    ["large IMDB"]="4:16:0"
                    ["small IMDB"]="1:00:0"    ["large IMDB"]="5:40:0"
                  # ["small MIMIC-a"]="0:05:0" ["large MIMIC-a"]="1:26:0"
                    ["small MIMIC-a"]="1:00:0" ["large MIMIC-a"]="1:50:0"
                  # ["small MIMIC-d"]="0:05:0" ["large MIMIC-d"]="2:32:0"
                    ["small MIMIC-d"]="1:30:0" ["large MIMIC-d"]="3:00:0"
                  # ["small MNLI"]="5:20:0"    ["large MNLI"]="10:30:0"
                    ["small MNLI"]="6:20:0"    ["large MNLI"]="11:30:0"
                  # ["small MRPC"]="0:11:0"    ["large MRPC"]="0:20:0"
                    ["small MRPC"]="0:35:0"    ["large MRPC"]="0:40:0"
                  # ["small QNLI"]="2:52:0"    ["large QNLI"]="6:28:0"
                    ["small QNLI"]="3:30:0"    ["large QNLI"]="7:00:0"
                  # ["small QQP"]="3:15:0"     ["large QQP"]="8:05:0"
                    ["small QQP"]="4:00:0"     ["large QQP"]="9:00:0"
                  # ["small RTE"]="0:25:0"     ["large RTE"]="0:34:0"
                    ["small RTE"]="0:55:0"     ["large RTE"]="0:55:0"
                  # ["small SST2"]="0:49:0"    ["large SST2"]="1:46:0"
                    ["small SST2"]="1:20:0"    ["large SST2"]="2:00:0"
                  # ["small WNLI"]="0:07:0"    ["large WNLI"]="0:10:0"
                    ["small WNLI"]="0:30:0"    ["large WNLI"]="0:30:0"  )

for model in 'roberta-sb' 'roberta-sl' # 'roberta-m15' 'roberta-m20' 'roberta-m30' 'roberta-m40' 'roberta-m50'
do
    for dataset in 'bAbI-1' 'bAbI-2' 'bAbI-3' 'BoolQ' 'CB' 'CoLA' 'IMDB' 'MIMIC-a' 'MIMIC-d' 'MNLI' 'MRPC' 'QNLI' 'QQP' 'RTE' 'SST2' 'WNLI'
    do
        sbatch --time=1:00:00 -J "masked-data-table_d-${dataset}_m-${model}" \
            -o "$SCRATCH"/ecoroar/logs/%x.%j.out -e "$SCRATCH"/ecoroar/logs/%x.%j.err \
            $(job_script gpu) \
            export/masked_data_table.py \
                --model "${model}" \
                --dataset "${dataset}"
    done
done
