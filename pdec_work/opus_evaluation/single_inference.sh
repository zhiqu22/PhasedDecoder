#!/bin/bash

METHOD=${1}
ID=${2}
src=${3}
tgt=${4}
gpu_id=${5}
ROOT_PATH=${6}
DATA_BIN=${7}
WORK_PATH=${ROOT_PATH}/pdec_work
SAVE_PATH=${WORK_PATH}/results


cd ${ROOT_PATH}/fairseq

task='translation_multi_simple_epoch'

DIR=""
if [ $METHOD == 'opus_pdec' ];then
  DIR='--user-dir models/PhasedDecoder/ '
fi

tgt_file=$src"-"$tgt".raw.txt"
CUDA_VISIBLE_DEVICES=$gpu_id, fairseq-generate ${DATA_BIN} --gen-subset test \
$DIR \
-s $src -t $tgt \
--lang-dict ${DATA_BIN}/languages.txt \
--lang-pairs ${DATA_BIN}/pairs.txt \
--path $WORK_PATH/checkpoints/$METHOD/$ID/checkpoint_best.pt \
--remove-bpe sentencepiece \
--required-batch-size-multiple 1 \
--skip-invalid-size-inputs-valid-test \
--task $task \
--encoder-langtok tgt \
--beam 4 > $SAVE_PATH/$METHOD/$ID/$tgt_file

# hypothesis
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h"
# reference
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r"
# source
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^S" | sort -t '-' -k2n | cut -f 2- | sed 's/__[a-zA-Z_]*__ //' > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s"
rm $SAVE_PATH/$METHOD/$ID/$tgt_file

TOK='13a'
if [ $tgt == 'zh' ];then
    TOK='zh'
fi
if [ $tgt == 'ja' ];then
    TOK='ja-mecab'
fi
if [ $tgt == 'ko' ];then
    TOK='ko-mecab'
fi


output=$(sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h" -w 4 -tok $TOK < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r")
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"
echo $output >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"

output=$(sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h" -w 4 -m chrf --chrf-word-order 2 < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r")
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"
echo $output >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"