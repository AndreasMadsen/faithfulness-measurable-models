
PROJECT_RESULT_DIR="${SCRATCH}/ecoroar"
mkdir -p "$PROJECT_RESULT_DIR"/logs

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

function multiply_time {
    local walltime=$1
    local multiplier=$2

    python3 -c \
        "from datetime import timedelta; \
        in_h, in_m = '${walltime}'.split(':'); \
        t = timedelta(hours=int(in_h), minutes=int(in_m)) * ${multiplier}; \
        out_h, out_m = divmod(int(t.total_seconds()) // 60, 60); \
        print(f'{out_h:d}:{out_m:d}')"
}

submit_seeds () {
    local walltime=$1
    local seeds=$2
    local experiment_name;

    if ! experiment_name=$(python experiments/experiment-name.py "${@:4}" --seed 9999); then
        echo -e "\e[31mCould not get experiment name, error ^^^${experiment_name}\e[0m" >&2
        return 1
    fi

    if [[ $walltime == *"?"* ]]; then
        echo -e "\e[33mUndefined walltime $walltime for ${experiment_name/9999/x}\e[0m" >&2
        return 0
    fi

    local run_seeds=()
    local filename
    for seed in $(echo "$seeds")
    do
        if [ ! -f "${PROJECT_RESULT_DIR}/results/${experiment_name%%_*}/${experiment_name/9999/$seed}.json" ]; then
            run_seeds+=($seed)
            echo "scheduling ${experiment_name/9999/$seed}" 1>&2
        fi
    done

    if [ ! "${#run_seeds[@]}" -eq 0 ]; then
        local walltime_times_nb_seeds;
        if ! walltime_times_nb_seeds=$(multiply_time "${walltime}" "${#run_seeds[@]}"); then
            echo -e "\e[31mCould not parse time '${walltime}', error ^^^${walltime_times_nb_seeds}\e[0m" >&2
            return 1
        fi

        local concat_seeds=$(join_by '' "${run_seeds[@]}")
        local jobname="${experiment_name/9999/$concat_seeds}"
        local jobid;
        if jobid=$(
            sbatch --time="$walltime_times_nb_seeds:0" \
               --parsable \
               --export=ALL,RUN_SEEDS="$(join_by ' ' "${run_seeds[@]}")" \
               -J "$jobname" \
               -o "$PROJECT_RESULT_DIR"/logs/%x.%j.out -e "$PROJECT_RESULT_DIR"/logs/%x.%j.err \
               "${@:3}"
        ); then
            echo -e "\e[32msubmitted job as ${jobid}\e[0m" >&2
        else
            echo -e "\e[31mCould not submit ${jobname}, error ^^^${jobid}\e[0m" >&2
            return 1
        fi
    else
        echo -e "\e[34mskipping ${experiment_name/9999/x}\e[0m" >&2
    fi
}
