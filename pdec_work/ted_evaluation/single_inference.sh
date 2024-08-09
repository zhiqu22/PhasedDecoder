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
DETOKENIZER=${ROOT_PATH}/moses/scripts/tokenizer/detokenizer.perl

cd ${ROOT_PATH}/fairseq

DIR=""
task='translation_multi_simple_epoch'
if [ $METHOD == 'ted_pdec' ];then
  DIR='--user-dir models/PhasedDecoder/ '
fi

tgt_file=$src"-"$tgt".raw.txt"
CUDA_VISIBLE_DEVICES=$gpu_id, fairseq-generate $WORK_PATH/datasets/ted_19-bin/ --gen-subset test \
$DIR \
-s $src -t $tgt \
--langs "en,ar,he,ru,ko,it,ja,zh,es,nl,vi,tr,fr,pl,ro,fa,hr,cs,de" \
--lang-pairs "ar-en,en-ar,he-en,en-he,ru-en,en-ru,ko-en,en-ko,it-en,en-it,ja-en,en-ja,zh-en,en-zh,es-en,en-es,nl-en,en-nl,vi-en,en-vi,tr-en,en-tr,fr-en,en-fr,pl-en,en-pl,ro-en,en-ro,fa-en,en-fa,hr-en,en-hr,cs-en,en-cs,de-en,en-de" \
--path $WORK_PATH/checkpoints/$METHOD/$ID/checkpoint_averaged.pt \
--remove-bpe sentencepiece \
--required-batch-size-multiple 1 \
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

cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h" | perl ${DETOKENIZER} -threads 32 -l $tgt >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r" | perl ${DETOKENIZER} -threads 32 -l $tgt >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s" | perl ${DETOKENIZER} -threads 32 -l $src >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.s"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s"

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

output=$(sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h" -w 4 -tok $TOK < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r")
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"
echo $output >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"

output=$(sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h" -w 4 -m chrf --chrf-word-order 2 < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r")
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"
echo $output >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"