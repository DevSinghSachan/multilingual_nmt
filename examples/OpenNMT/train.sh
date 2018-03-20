# Train Transformer Model
python train.py -gpuid 0 -save_model models_ja_en -data data/demo -layers 1 -rnn_size 512 -word_vec_size 512 \
-epochs 30 -max_grad_norm 0 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding \
-dropout 0.2 -param_init 0 -warmup_steps 4000 -learning_rate 1.0 -decay_method noam -label_smoothing 0.0 \
-adam_beta2 0.98 -batch_size 100 -start_decay_at 31


# Train BiLSTM Model
python train.py -gpuid 0 -save_model models_ja_en -data data/demo -layers 1 -rnn_size 512 -word_vec_size 512 \
-epochs 30 -optim adam -encoder_type brnn -decoder_type rnn -dropout 0.2 -learning_rate 0.001  -batch_size 100