#!/bin/bash

DATA_BIN="."
ROOT_PATH="."
METHOD=opus_pdec
ID=1


ADAPTION='True'
DROP=0.1
INNER=4096
BIAS=1
CONTRASTIVE='True'
TYPE='enc'
POSITION=6
T=1.0
DIM=1024
MODE=1
SEED=0

FAIR_PATH=${ROOT_PATH}/fairseq
WORK_PATH=${ROOT_PATH}/pdec_work
mv ${WORK_PATH}/train/languages.txt ${DATA_BIN}/languages.txt 
mv ${WORK_PATH}/train/pairs.txt ${DATA_BIN}/pairs.txt 
cd ${FAIR_PATH}

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
--seed $SEED --fp16 --ddp-backend=no_c10d --arch transformer_pdec_big_1024 --task translation_multi_simple_epoch \
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
--lang-dict ${DATA_BIN}/languages.txt \
--lang-pairs ${DATA_BIN}/pairs.txt \
--sampling-method "temperature" --sampling-temperature 5 \
--encoder-langtok tgt \
--encoder-normalize-before --decoder-normalize-before \
--max-source-positions 128 --max-target-positions 128 \
--skip-invalid-size-inputs-valid-test \
--criterion label_smoothed_cross_entropy_instruction --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0007 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --max-update 50000 --max-tokens 4000 --update-freq 16 \
--share-all-embeddings --weight-decay 0.0 \
--no-epoch-checkpoints --no-progress-bar \
--dropout 0.1 \
--keep-best-checkpoints 1 --log-interval 100 --log-format simple \
--save-dir ${WORK_PATH}/checkpoints/${METHOD}/${ID} > ${WORK_PATH}/logs/${METHOD}/${ID}.log


cd ${WORK_PATH}
bash opus_evaluation/batch_inference.sh ${METHOD} ${ID} ${ROOT_PATH} ${num_gpus} ${DATA_BIN}
# after inference, compute bs and comet
python opus_evaluation/opus_bertscore.py $NAME $ID '2en' 0 $WORK_PATH &
python opus_evaluation/opus_bertscore.py $NAME $ID 'en2' 1 $WORK_PATH &
python opus_evaluation/opus_bertscore.py $NAME $ID 'zero' 2 $WORK_PATH &
CUDA_VISIBLE_DEVICES=3 python opus_evaluation/opus_comet.py $NAME $ID '2en' $WORK_PATH &
CUDA_VISIBLE_DEVICES=4 python opus_evaluation/opus_comet.py $NAME $ID 'en2' $WORK_PATH &
CUDA_VISIBLE_DEVICES=5 python opus_evaluation/opus_comet.py $NAME $ID 'zero' $WORK_PATH &
wait

mkdir ${WORK_PATH}/excel
python opus_evaluation/make_table.py ${METHOD} ${ID} ${ROOT_PATH}