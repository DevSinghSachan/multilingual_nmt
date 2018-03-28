#!/usr/bin/env bash

TF=$(pwd)
export PATH=$PATH:$TF/bin

BPE_OPS=16000
GPUARG=0

L1=$1
L2=$2
L3=$3
model=$4

DATA_L1=${TF}"/data/${L1}_${L2}"
DATA_L2=${TF}"/data/${L1}_${L3}"
NAME="run_${L1}_${L2}-${L3}"
OUT="temp/$NAME"

TEST_SRC_L1=$DATA_L1/test.${L1}
TEST_TGT_L1=$DATA_L1/test.${L2}

TEST_SRC_L2=$DATA_L2/test.${L1}
TEST_TGT_L2=$DATA_L2/test.${L3}

# Apply BPE Coding to the languages
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < ${TEST_SRC_L1} > ${OUT}/data/test_l1.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < ${TEST_SRC_L2} > ${OUT}/data/test_l2.src


# Translate Language 1
python translate.py -i $OUT/data --data processed --batchsize 28 --beam_size 5 \
--best_model_file $OUT/models/model_best_$NAME.ckpt --src $OUT/data/test_l1.src \
--gpu $GPUARG --output $OUT/test/test_l1.out --model ${model}


# Translate Language 2
python translate.py -i $OUT/data --data processed --batchsize 28 --beam_size 5 \
--best_model_file $OUT/models/model_best_$NAME.ckpt --src $OUT/data/test_l2.src \
--gpu $GPUARG --output $OUT/test/test_l2.out --model ${model}


mv $OUT/test/test_l1.out{,.bpe}
mv $OUT/test/test_l2.out{,.bpe}


cat $OUT/test/test_l1.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test_l1.out
cat $OUT/test/test_l2.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test_l2.out


perl tools/multi-bleu.perl $TEST_TGT_L1 < $OUT/test/test_l1.out > $OUT/test/test_l1.tc.bleu
perl tools/multi-bleu.perl $TEST_TGT_L2 < $OUT/test/test_l2.out > $OUT/test/test_l2.tc.bleu
