import os
import pickle
import torch

import preprocess
from models import Transformer
from translate import TranslateText


model_path = {
    'ja->en': {
        'sent_enc': 'temp/run_ja_en_wpm_large/data/wpm-codes.32000.model',
        'vocab_path': 'temp/run_ja_en_wpm_large/data/processed.vocab.pickle',
        'model_path':'temp/run_ja_en_wpm_large/models/model_best_run_ja_en_wpm_large.ckpt'
    },
    'en->ja': {
        'sent_enc':'temp/run_en_ja_wpm_large/data/wpm-codes.32000.model',
        'vocab_path': 'temp/run_en_ja_wpm_large/data/processed.vocab.pickle',
        'model_path': 'temp/run_en_ja_wpm_large/models/model_best_run_en_ja_wpm_large.ckpt'
    },
    'en->fr': 'temp/run_wmt14_en_fr_large/models/model_best_run_wmt14_en_fr_large.ckpt',
    'fr->en': 'temp/run_wmt14_en_fr_large/models/model_best_run_wmt14_en_fr_large.ckpt',
    'en->de': '',
    'de->en': ''
}


def translate(sent, src='ja', tgt='en'):
    key = '{}->{}'.format(src, tgt)

    # Encode the sentence
    if src == 'ja' or tgt == 'ja':
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path[key]['sent_enc'])
        sent = ' '.join(map(str, sp.EncodeAsIds(sent.strip())))
        fp = open('/tmp/tmp.txt', 'w')
        fp.write(sent + '\n')
        fp.close()

    # Read the vocab file
    with open(model_path[key]['vocab_path'], 'rb') as f:
        id2w = pickle.load(f)
    w2id = {w: i for i, w in id2w.items()}
    source_data = preprocess.make_dataset(os.path.realpath('/tmp/tmp.txt'), w2id)

    # Translation code
    checkpoint = torch.load(model_path[key]['model_path'])
    config = checkpoint['opts']
    model = Transformer(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    # Translating State Dict
    hyp = TranslateText(model,
                        source_data,
                        batch=2,
                        beam_size=4,
                        alpha=0.6,
                        max_decode_len=80)()

    sent_list = []
    for sent in hyp:
        words = [id2w[y] for y in sent]
        words = list(map(int, words))
        sent = sp.DecodeIds(words)
        sent_list.append(sent)

    return sent_list


