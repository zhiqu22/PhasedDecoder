#!/bin/bash

METHOD=${1}
ID=${2}
ROOT_PATH=${3}
num_gpus=${4}
DATA_BIN=${5}

WORK_PATH=${ROOT_PATH}/pdec_work
SAVE_PATH=${WORK_PATH}/results
cd ${WORK_PATH}

pids=()
declare -A lang_pairs

mkdir $SAVE_PATH/$METHOD/${ID}
for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    src="en"
    tgt=$lang
    lang_pairs["$src,$tgt"]=1
    src=$lang
    tgt="en"
    lang_pairs["$src,$tgt"]=1
done
for lpair in de-nl nl-zh ar-nl ru-zh fr-nl de-fr fr-zh ar-ru ar-zh ar-fr de-zh fr-ru de-ru nl-ru ar-de; do
    TMP=(${lpair//-/ })
    src=${TMP[0]}
    tgt=${TMP[1]}
    lang_pairs["$src,$tgt"]=1
    src=${TMP[1]}
    tgt=${TMP[0]}
    lang_pairs["$src,$tgt"]=1
done

for gpu_id in $(seq 0 $((num_gpus - 1))); do
    if [[ ${#lang_pairs[@]} -gt 0 ]]; then
        IFS=',' read -r src tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
        unset lang_pairs["$src,$tgt"]
        bash opus_evaluation/single_inference.sh $METHOD $ID $mode $src $tgt $gpu_id $work_path &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            IFS=',' read -r src tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$src,$tgt"]
            bash opus_evaluation/single_inference.sh $METHOD $ID $mode $src $tgt $gpu_id $work_path &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break
    fi
    sleep 5
done

wait
echo "All translations completed."