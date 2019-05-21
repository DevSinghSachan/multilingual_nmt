#!/usr/bin/env bash

TF=$(pwd)
export PATH=$PATH:${TF}/bin

NAME="run_wmt16_de_en"
OUT="temp/$NAME"

model_file=$1

mkdir -p /tmp/$NAME/tmp
valid_out=/tmp/$NAME/tmp/valid_${model_file}.out

mkdir -p /tmp/$NAME/result
results_file=/tmp/$NAME/result/${model_file}

python translate.py -i $OUT/data --data processed --batchsize 80 --beam_size 4 \
--best_model_file $OUT/models/${model_file} --src $OUT/data/valid.src \
--gpu 0 --output ${valid_out} --alpha 0.6 --max_decode_len 80

echo "Default BLEU" >> ${results_file}
mv ${valid_out}{,.bpe}
cat ${valid_out}.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > ${valid_out}
t2t-bleu --translation=${valid_out} --reference=/results/wmt16_de_en/newstest2014.tok.de >> ${results_file}

echo "EMA BLEU" >> ${results_file}
mv ${valid_out}.ema{,.bpe}
cat ${valid_out}.ema.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > ${valid_out}.ema
t2t-bleu --translation=${valid_out}.ema --reference=/results/wmt16_de_en/newstest2014.tok.de >> ${results_file}

