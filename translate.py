# encoding: utf-8
from __future__ import unicode_literals, print_function

import os
import pickle
import json
import torch
from tqdm import tqdm

from models import MultiTaskNMT, Transformer
import utils
from torch.autograd import Variable
import preprocess
from train import save_output
from config import get_translate_args


class TranslateText(object):
    def __init__(self, model, test_data, batch=50, max_length=50, beam_size=1,
                 alpha=1.0):
        self.model = model
        self.test_data = test_data
        self.batch = batch
        self.device = -1
        self.max_length = max_length
        self.beam_size = beam_size
        self.alpha = alpha

    def __call__(self):
        self.model.eval()
        hypotheses = []
        for i in tqdm(range(0, len(self.test_data), self.batch)):
            sources = self.test_data[i:i + self.batch]
            x_block = utils.source_pad_concat_convert(sources,
                                                      device=None)
            x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE),
                               requires_grad=False)
            ys = self.model.translate(x_block,
                                      self.max_length,
                                      beam=self.beam_size,
                                      alpha=self.alpha)
            hypotheses.extend(ys)
        return hypotheses


def main():
    args = get_translate_args()
    print(json.dumps(args.__dict__, indent=4))

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'),
              'rb') as f:
        id2w = pickle.load(f)

    w2id = {w: i for i, w in id2w.items()}
    source_data = preprocess.make_dataset(os.path.realpath(args.src),
                                          w2id,
                                          args.tok)

    checkpoint = torch.load(args.best_model_file)
    print("=> loaded checkpoint '{}' (epoch {}, best score {})".
          format(args.best_model_file,
                 checkpoint['epoch'],
                 checkpoint['best_score']))
    config = checkpoint['opts']
    model = eval(args.model)(config)
    model.load_state_dict(checkpoint['state_dict'])

    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    hyp = TranslateText(model,
                        source_data,
                        batch=args.batchsize // 4,
                        beam_size=args.beam_size,
                        alpha=args.alpha,
                        max_length=args.max_len)()
    save_output(hyp, id2w, args.output)


if __name__ == '__main__':
    main()
