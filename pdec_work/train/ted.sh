#!/bin/bash

# binary dataset path
DATA_BIN="."
ROOT_PATH="."
num_gpus=8
FAIR_PATH=${ROOT_PATH}/fairseq
WORK_PATH=${ROOT_PATH}/pdec_work
cd ${FAIR_PATH}

METHOD=ted_pdec
ID=1

ENC=6
DEC=6
BIAS=1
ADAPTION='True'
DROP=0.1
INNER=2048
CONTRASTIVE='True'
POSITION=6
TYPE='enc'
T=1.0
DIM=512
MODE=1

SEED=0

mkdir ${WORK_PATH}/checkpoints
mkdir ${WORK_PATH}/checkpoints/${METHOD}
mkdir ${WORK_PATH}/checkpoints/${METHOD}/${ID}

mkdir ${WORK_PATH}/logs
mkdir ${WORK_PATH}/logs/${METHOD}

mkdir ${WORK_PATH}/results
mkdir ${WORK_PATH}/results/${METHOD}
mkdir ${WORK_PATH}/results/${METHOD}/${ID}



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ${DATA_BIN} \
--user-dir models/PhasedDecoder/ \
--seed $SEED --fp16 --ddp-backend=no_c10d --arch transformer_pdec_${ENC}_e_${DEC}_d --task translation_multi_simple_epoch \
--sampling-method "temperature" --sampling-temperature 5 \
--attention-position-bias $BIAS \
--adaption-flag $ADAPTION \
--adaption-inner-size $INNER \
--adaption-dropout $DROP \
--contrastive-flag $CONTRASTIVE \
--contrastive-type $TYPE \
--dim $DIM \
--mode $MODE \
--cl-position $POSITION \
--temperature $T \
--langs "en,ar,he,ru,ko,it,ja,zh,es,nl,vi,tr,fr,pl,ro,fa,hr,cs,de" \
--lang-pairs "ar-en,en-ar,he-en,en-he,ru-en,en-ru,ko-en,en-ko,it-en,en-it,ja-en,en-ja,zh-en,en-zh,es-en,en-es,nl-en,en-nl,vi-en,en-vi,tr-en,en-tr,fr-en,en-fr,pl-en,en-pl,ro-en,en-ro,fa-en,en-fa,hr-en,en-hr,cs-en,en-cs,de-en,en-de" \
--encoder-langtok tgt \
--criterion label_smoothed_cross_entropy_instruction --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --max-epoch 30 --max-tokens 4000 \
--share-all-embeddings --weight-decay 0.0001 \
--no-epoch-checkpoints --no-progress-bar \
--keep-best-checkpoints 5 --log-interval 1000 --log-format simple \
--save-dir ${WORK_PATH}/checkpoints/${METHOD}/${ID} > ${WORK_PATH}/logs/${METHOD}/${ID}.log

checkpoints=$(ls ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint.best_loss_* | tr '\n' ' ')
python3 scripts/average_checkpoints.py \
--inputs $checkpoints \
--output ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt

cd ${WORK_PATH}
bash ted_evaluation/batch_inference.sh ${METHOD} ${ID} ${ROOT_PATH} ${num_gpus} ${DATA_BIN}
bash ted_evaluation/batch_bertscore.sh ${METHOD} ${ID} ${WORK_PATH} ${num_gpus}
bash ted_evaluation/batch_comet.sh ${METHOD} ${ID} ${WORK_PATH} ${num_gpus}

mkdir ${WORK_PATH}/excel
python ted_evaluation/make_table.py ${METHOD} ${ID} ${ROOT_PATH}