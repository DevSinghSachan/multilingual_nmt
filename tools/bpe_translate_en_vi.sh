#!/usr/bin/env bash

TF=$(pwd)
export PATH=$TF/bin:$PATH

BPE_OPS=32000
GPUARG=0

optimizer=$1
lr=$2
beta1=$3
beta2=$4
eps=$5

NAME="run_en_vi_${optimizer}_${lr}_${beta1}_${beta2}_${eps}"
OUT="temp/$NAME"

TEST_SRC=$OUT/data/test.src
TEST_TGT=$OUT/data/test.tgt

# Apply BPE Coding to the languages
# apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < ${TEST_SRC} > ${OUT}/data/test.src

# Translate
python translate.py -i $OUT/data --data processed --batchsize 28 --beam_size 5 \
--best_model_file $OUT/models/model_best_$NAME.ckpt --src $OUT/data/test.src \
--gpu $GPUARG --output $OUT/test/test.out --model Transformer --max_decode_len 70


mv $OUT/test/test.out $OUT/test/test.out.bpe
cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

perl tools/multi-bleu.perl $TEST_TGT < $OUT/test/test.out > $OUT/test/test.tc.bleu
t2t-bleu --translation=$OUT/test/test.out --reference=$TEST_TGT > $OUT/test/test.t2t-bleu


mv $OUT/test/test.out.ema $OUT/test/test.out.bpe.ema
cat $OUT/test/test.out.bpe.ema | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out.ema

perl tools/multi-bleu.perl $TEST_TGT < $OUT/test/test.out.ema > $OUT/test/test.tc.bleu.ema
t2t-bleu --translation=$OUT/test/test.out.ema --reference=$TEST_TGT > $OUT/test/test.t2t-bleu.ema

