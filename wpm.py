import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=data/en_vi/train.en --model_prefix=m --vocab_size=8000')

