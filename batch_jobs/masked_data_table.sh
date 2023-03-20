source "batch_jobs/_job_script.sh"

declare -A size=( ["roberta-sb"]="small" ["roberta-sl"]="large"
                  ["roberta-m15"]="large" ["roberta-m20"]="large" ["roberta-m30"]="large" ["roberta-m40"]="large" ["roberta-m50"]="large" )

#                   V100                       V100
declare -A time=( # ["small bAbI-1"]="0:06"  ["large bAbI-1"]="0:44"
                    ["small bAbI-1"]="0:15"  ["large bAbI-1"]="1:00"
                  # ["small bAbI-2"]="0:06"  ["large bAbI-2"]="1:14"
                    ["small bAbI-2"]="0:15"  ["large bAbI-2"]="1:30"
                  # ["small bAbI-3"]="0:06"  ["large bAbI-3"]="2:29"
                    ["small bAbI-3"]="0:15"  ["large bAbI-3"]="3:00"
                  # ["small BoolQ"]="0:05"   ["large BoolQ"]="1:16"
                    ["small BoolQ"]="1:00"   ["large BoolQ"]="1:50"
                  # ["small CB"]="0:05"      ["large CB"]="0:17"
                    ["small CB"]="0:50"      ["large CB"]="0:50"
                  # ["small CoLA"]="0:13"    ["large CoLA"]="2:33"
                    ["small CoLA"]="1:40"    ["large CoLA"]="3:00"
                  # ["small IMDB"]="0:44"    ["large IMDB"]="4:16"
                    ["small IMDB"]="1:00"    ["large IMDB"]="5:40"
                  # ["small MIMIC-a"]="0:05" ["large MIMIC-a"]="1:26"
                    ["small MIMIC-a"]="1:00" ["large MIMIC-a"]="1:50"
                  # ["small MIMIC-d"]="0:05" ["large MIMIC-d"]="2:32"
                    ["small MIMIC-d"]="1:30" ["large MIMIC-d"]="3:00"
                  # ["small MNLI"]="5:20"    ["large MNLI"]="10:30"
                    ["small MNLI"]="6:20"    ["large MNLI"]="11:30"
                  # ["small MRPC"]="0:11"    ["large MRPC"]="0:20"
                    ["small MRPC"]="0:35"    ["large MRPC"]="0:40"
                  # ["small QNLI"]="2:52"    ["large QNLI"]="6:28"
                    ["small QNLI"]="3:30"    ["large QNLI"]="7:00"
                  # ["small QQP"]="3:15"     ["large QQP"]="8:05"
                    ["small QQP"]="4:00"     ["large QQP"]="9:00"
                  # ["small RTE"]="0:25"     ["large RTE"]="0:34"
                    ["small RTE"]="0:55"     ["large RTE"]="0:55"
                  # ["small SST2"]="0:49"    ["large SST2"]="1:46"
                    ["small SST2"]="1:20"    ["large SST2"]="2:00"
                  # ["small WNLI"]="0:07"    ["large WNLI"]="0:10"
                    ["small WNLI"]="0:30"    ["large WNLI"]="0:30"  )

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
