#!/bin/bash


METHOD=${1}
ID=${2}
WORK_PATH=${3}
num_gpus=${4}

pids=()
declare -A lang_pairs

for tgt in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
    lang_pairs["$tgt"]=1
done

for gpu_id in $(seq 0 $((num_gpus - 1))); do
    if [[ ${#lang_pairs[@]} -gt 0 ]]; then
        IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
        unset lang_pairs["$tgt"]
        python ted_evaluation/ted_bertscore.py $METHOD $ID $tgt $gpu_id $WORK_PATH &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            IFS=',' read -r tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$tgt"]
            python ted_evaluation/ted_bertscore.py $METHOD $ID $tgt $gpu_id $WORK_PATH &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break
    fi
    sleep 5
done

wait
echo "All evaluations completed."