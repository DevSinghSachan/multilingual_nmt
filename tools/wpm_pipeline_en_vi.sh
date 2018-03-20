#!/usr/bin/env bash

TF=$(pwd)

export PATH=$PATH:$TF/bin
#======= EXPERIMENT SETUP ======

# update these variables
NAME="run_en_vi_wpm"
OUT="temp/$NAME"

DATA=${TF}"/data/en_vi"
TRAIN_SRC=$DATA/train.en
TRAIN_TGT=$DATA/train.vi
TEST_SRC=$DATA/tst2013.en
TEST_TGT=$DATA/tst2013.vi
VALID_SRC=$DATA/tst2012.en
VALID_TGT=$DATA/tst2012.vi

WPM="src+tgt" # src, tgt, src+tgt
VOCAB_SIZE=16000
GPUARG="0"

#====== EXPERIMENT BEGIN ======

echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test


echo "Step 1a: Preprocess inputs"

echo "Learning Word Piece on source and target combined"
spm_train --input=${TRAIN_SRC},${TRAIN_TGT} --vocab_size ${VOCAB_SIZE} --model_prefix=$OUT/data/wpm-codes.${VOCAB_SIZE}

echo "Applying Word Piece on source"
spm_encode --model $OUT/data/wpm-codes.${VOCAB_SIZE}.model --output_format=id < $TRAIN_SRC > $OUT/data/train.src
spm_encode --model $OUT/data/wpm-codes.${VOCAB_SIZE}.model --output_format=id < $VALID_SRC > $OUT/data/valid.src
spm_encode --model $OUT/data/wpm-codes.${VOCAB_SIZE}.model --output_format=id < $TEST_SRC > $OUT/data/test.src

echo "Applying Word Piece on target"
spm_encode --model $OUT/data/wpm-codes.${VOCAB_SIZE}.model --output_format=id <  $TRAIN_TGT > $OUT/data/train.tgt
spm_encode --model $OUT/data/wpm-codes.${VOCAB_SIZE}.model --output_format=id <  $VALID_TGT > $OUT/data/valid.tgt
# We dont touch the test References, No BPE on them!
cp $TEST_TGT $OUT/data/test.tgt


echo "Step 1b: Preprocess"
python ${TF}/preprocess.py -i ${OUT}/data \
      -s-train train.src \
      -t-train train.tgt \
      -s-valid valid.src \
      -t-valid valid.tgt \
      -s-test test.src \
      -t-test test.tgt \
      --save_data processed


echo "Step 2: Train"
CMD="python $TF/train.py -i $OUT/data --data processed --model_file $OUT/models/model_$NAME.ckpt --data processed \
--batchsize 60 --tied --beam_size 5 --epoch 40 --layers 6 --multi_heads 8 --gpu $GPUARG --dev_hyp $OUT/test/valid.out \
--test_hyp $OUT/test/test.out"

echo "Training command :: $CMD"
eval "$CMD"


# select a model with high accuracy and low perplexity
model=$OUT/models/model_$NAME.ckpt
echo "Chosen Model = $model"
if [[ -z "$model" ]]; then
    echo "Model not found. Looked in $OUT/models/"
    exit 1
fi

echo "WPM decoding/detokenising target to match with references"
mv $OUT/test/test.out{,.wpm}
mv $OUT/test/valid.out{,.wpm}
spm_decode --model ${OUT}/data/wpm-codes.${VOCAB_SIZE}.model --input_format=id < ${OUT}/test/valid.out.wpm > $OUT/test/valid.out
spm_decode --model ${OUT}/data/wpm-codes.${VOCAB_SIZE}.model --input_format=id < ${OUT}/test/test.out.wpm > $OUT/test/test.out

echo "Step 4a: Evaluate Test"
perl $TF/tools/multi-bleu.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
perl $TF/tools/multi-bleu.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
