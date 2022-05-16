
job_script () {
    local loginnode=${HOSTNAME%%.*}
    local cluster=${loginnode//[0-9]/}
    local jobscript="python_${cluster}_$1_job.sh"

    if [ ! -f "$jobscript" ]; then
        echo "python_${cluster}_$1_job.sh not found" 1>&2
        return 1
    fi

    echo "$jobscript"
}

function join_by {
    local IFS="$1";
    shift;
    echo "$*";
}

submit_seeds () {
    local walltime=$1
    local seeds=$2
    local experiment_name=$(python "${@:4}" --experiment-name --seed 9999)

    local run_seeds=()
    local filename
    for seed in $(echo "$seeds")
    do
        if [ ! -f "${SCRATCH}/nlproar/results/${experiment_name/9999/"$seed"}.json" ]; then
            run_seeds+=($seed)
            echo "scheduling $experiment_name" 1>&2
        fi
    done

    if [ ! "${#run_seeds[@]}" -eq 0 ]; then
        local walltime_times_nb_seeds=$(python3 -c \
        "from datetime import datetime; \
         t = (datetime.strptime('$walltime', '%H:%M:%S') - datetime.strptime('0:0:0', '%H:%M:%S')) * ${#run_seeds[@]}; \
         print(':'.join(map(str, [*divmod(int(t.total_seconds()) // 60, 60), 0])));
        ")

        local concat_seeds=$(join_by '' "${run_seeds[@]}")
        local jobname="${experiment_name/9999/"$concat_seeds"}"
        sbatch --time="$walltime_times_nb_seeds" \
               --export=ALL,RUN_SEEDS="$(join_by ' ' "${run_seeds[@]}")" \
               -J "$jobname" \
               -o "$SCRATCH"/ecoroar/logs/%x.%j.out -e "$SCRATCH"/ecoroar/logs/%x.%j.err \
               "${@:3}"
    else
        echo "skipping"
    fi
}
