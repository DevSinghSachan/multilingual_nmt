#!/usr/bin/env bash

TF=$(pwd)

export PATH=$PATH:$TF/bin
#======= EXPERIMENT SETUP ======

# update these variables
NAME="run_wmt14_en_fr_large"
OUT="temp/$NAME"

DATA="data/wmt14_enfr"
TRAIN_SRC=$DATA/train.en
TRAIN_TGT=$DATA/train.fr
TEST_SRC=$DATA/valid.en
TEST_TGT=$DATA/valid.fr
VALID_SRC=$DATA/test.en
VALID_TGT=$DATA/test.fr

BPE="src+tgt" # src, tgt, src+tgt
BPE_OPS=32000
GPUARG="0"
#====== EXPERIMENT BEGIN ======

echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test
<<COMMENT
echo "Step 1a: Preprocess inputs"

echo "Learning BPE on source and target combined"
cat ${TRAIN_SRC} ${TRAIN_TGT} | learn_bpe -s ${BPE_OPS} > $OUT/data/bpe-codes.${BPE_OPS}

echo "Applying BPE on source"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TRAIN_SRC > $OUT/data/train.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $VALID_SRC > $OUT/data/valid.src
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} < $TEST_SRC > $OUT/data/test.src

echo "Applying BPE on target"
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_TGT > $OUT/data/train.tgt
apply_bpe -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_TGT > $OUT/data/valid.tgt
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
      --save_data processed \
      --max_seq_len 80
COMMENT

echo "Step 2: Train"
CMD="python -m pdb $TF/train.py -i $OUT/data --data processed \
--model_file $OUT/models/model_$NAME.ckpt --best_model_file $OUT/models/model_best_$NAME.ckpt \
--batchsize 30 --tied --beam_size 4 --alpha 0.6 --epoch 10 \
--layers 6 --multi_heads 16 --gpu $GPUARG --max_decode_len 80 \
--dev_hyp $OUT/test/valid.out --test_hyp $OUT/test/test.out \
--metric bleu --wbatchsize 3000 --model Transformer \
--grad_accumulator_count 8 --warmup_steps 8000 --eval_steps 5000 --fp16 --dynamic_loss_scale --label_smoothing 0.1 \
--n_units 1024 --n_hidden 4096 --log_path $OUT/${NAME}.log"

echo "Training command :: $CMD"
eval "$CMD"

# select a model with high accuracy and low perplexity
model=$OUT/models/model_$NAME.ckpt
echo "Chosen Model = $model"
if [[ -z "$model" ]]; then
    echo "Model not found. Looked in $OUT/models/"
    exit 1
fi


echo "BPE decoding/detokenising target to match with references"
mv $OUT/test/test.out{,.bpe}
mv $OUT/test/valid.out{,.bpe}
cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

echo "Step 4a: Evaluate Test"
perl $TF/tools/multi-bleu.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
perl $TF/tools/multi-bleu.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
t2t-bleu --translation=$OUT/test/test.out --reference=$OUT/data/test.tgt



exit
# Secondary commands to translate using translate.py
NAME="run_wmt16_de_en"
OUT="/storage/devendra/temp/$NAME"
python translate.py -i $OUT/data --data processed --batchsize 28 --beam_size 4 \
--best_model_file $OUT/models/model_best_$NAME.ckpt --src $OUT/data/valid.src \
--gpu 0 --output $OUT/test/valid.out --alpha 0.6

mv $OUT/test/valid.out{,.bpe}
cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
t2t-bleu --translation=$OUT/test/valid.out --reference=/storage/devendra/temp/wmt16_de_en/newstest2014.tok.de
