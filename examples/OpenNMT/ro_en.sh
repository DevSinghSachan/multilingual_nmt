#!/usr/bin/env bash

DATA_DIR=../../Attention_is_All_You_Need/data

# Ro + Nl --> En

# Preprocess
python preprocess.py -train_src $DATA_DIR/train.ro-nl -train_tgt $DATA_DIR/train.en -valid_src $DATA_DIR/dev.ro-nl \
-valid_tgt $DATA_DIR/dev.en -save_data data/demo_ro-nl_en -src_vocab_size 40000 -tgt_vocab_size 40000

# Train the model
python train.py -gpuid 0 -save_model models_ro-nl_en -data data/demo_ro-nl_en -layers 1 -rnn_size 512 \
-word_vec_size 512 -epochs 30 -optim adam -encoder_type brnn -decoder_type rnn -dropout 0.2 -learning_rate 0.001  \
-batch_size 100

# Test the model on Ro
python translate.py -model models_ro-nl_en_acc_60.35_ppl_10.78_e30.pt -src $DATA_DIR/ro_en/test.ro \
-output pred.txt -verbose -report_bleu -gpu 0 -beam_size 5 -tgt $DATA_DIR/ro_en/test.en -alpha 1

# Test the model on Nl
python translate.py -model models_ro-nl_en_acc_60.35_ppl_10.78_e30.pt -src $DATA_DIR/nl_en/test.nl \
-output pred.txt -verbose -report_bleu -gpu 0 -beam_size 5 -tgt $DATA_DIR/nl_en/test.en -alpha 1



# Nl --> En

# Preprocess
python preprocess.py -train_src $DATA_DIR/nl_en/train.nl -train_tgt $DATA_DIR/nl_en/train.en \
-valid_src $DATA_DIR/nl_en/dev.nl -valid_tgt $DATA_DIR/nl_en/dev.en -save_data data/demo_nl_en \
-src_vocab_size 40000 -tgt_vocab_size 40000

# Train the model
python train.py -gpuid 0 -save_model models_nl_en -data data/demo_nl_en -layers 1 -rnn_size 512 -word_vec_size 512 \
-epochs 30 -optim adam -encoder_type brnn -decoder_type rnn -dropout 0.2 -learning_rate 0.001  -batch_size 100

# Test the model on Nl
python translate.py -model models_nl_en_acc_61.12_ppl_9.99_e30.pt -src $DATA_DIR/nl_en/test.nl \
-output pred.txt -verbose -report_bleu -gpu 0 -beam_size 5 -tgt $DATA_DIR/nl_en/test.en -alpha 1



# Ro --> En
python preprocess.py -train_src $DATA_DIR/ro_en/train.ro -train_tgt $DATA_DIR/ro_en/train.en \
-valid_src $DATA_DIR/ro_en/dev.ro -valid_tgt $DATA_DIR/ro_en/dev.en -save_data data/demo_ro_en \
-src_vocab_size 40000 -tgt_vocab_size 40000

python train.py -gpuid 0 -save_model models_ro_en -data data/demo_ro_en -layers 1 -rnn_size 512 -word_vec_size 512 \
-epochs 30 -optim adam -encoder_type brnn -decoder_type rnn -dropout 0.2 -learning_rate 0.001  -batch_size 100

python translate.py -model models_ro_en_acc_60.50_ppl_10.36_e30.pt -src $DATA_DIR/ro_en/test.ro \
-output pred.txt -verbose -report_bleu -gpu 0 -beam_size 5 -tgt $DATA_DIR/ro_en/test.en -alpha 1


# De --> En
python preprocess.py -train_src $DATA_DIR/de_en/train.de -train_tgt $DATA_DIR/de_en/train.en \
-valid_src $DATA_DIR/de_en/dev.de -valid_tgt $DATA_DIR/de_en/dev.en -save_data data/demo_de_en \
-src_vocab_size 40000 -tgt_vocab_size 40000

python train.py -gpuid 0 -save_model models_de_en -data data/demo_de_en -layers 1 -rnn_size 512 -word_vec_size 512 \
-epochs 30 -optim adam -encoder_type brnn -decoder_type rnn -dropout 0.2 -learning_rate 0.001  -batch_size 100

python translate.py -model models_de_en_acc_58.29_ppl_12.64_e30.pt -src $DATA_DIR/de_en/test.de \
-output pred.txt -verbose -report_bleu -gpu 0 -beam_size 5 -tgt $DATA_DIR/de_en/test.en -alpha 1