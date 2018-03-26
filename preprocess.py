from __future__ import unicode_literals
import collections
import io
import re
import six
import numpy as np
import progressbar
import json
import os
import pickle
from collections import namedtuple
from config import get_preprocess_args


split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')

Special_Seq = namedtuple('Special_Seq', ['PAD', 'EOS', 'UNK', 'BOS'])
Vocab_Pad = Special_Seq(PAD=0, EOS=1, UNK=2, BOS=3)


def split_sentence(s, tok=False):
    if tok:
        s = s.lower()
        s = s.replace('\u2019', "'")
        s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        if tok:
            words.extend(split_pattern.split(word))
        else:
            words.append(word)
    words = [w for w in words if w]
    return words


def open_file(path):
    return io.open(path, encoding='utf-8', errors='ignore')


def count_lines(path):
    with open_file(path) as f:
        return sum([1 for _ in f])


def read_file(path, tok=False):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with open_file(path) as f:
        for line in bar(f, max_value=n_lines):
            words = split_sentence(line, tok)
            yield words


def count_words(path, max_vocab_size=40000, tok=False):
    counts = collections.Counter()
    for words in read_file(path, tok):
        for word in words:
            counts[word] += 1

    vocab = [word for (word, _) in counts.most_common(max_vocab_size)]
    return vocab


def make_dataset(path, w2id, tok=False):
    # w2id = {word: index for index, word in enumerate(vocab)}
    dataset = []
    token_count = 0
    unknown_count = 0
    for words in read_file(path, tok):
        array = make_array(w2id, words)
        dataset.append(array)
        token_count += array.size
        unknown_count += (array == Vocab_Pad.UNK).sum()
    print('# of tokens: %d' % token_count)
    print('# of unknown: %d (%.2f %%)' % (unknown_count,
                                          100. * unknown_count / token_count))
    return dataset


def make_array(word_id, words):
    ids = [word_id.get(word, Vocab_Pad.UNK) for word in words]
    return np.array(ids, 'i')


if __name__ == "__main__":
    args = get_preprocess_args()

    print(json.dumps(args.__dict__, indent=4))

    # Vocab Construction
    source_path = os.path.join(args.input, args.source_train)
    target_path = os.path.join(args.input, args.target_train)

    src_cntr = count_words(source_path, args.source_vocab, args.tok)
    trg_cntr = count_words(target_path, args.target_vocab, args.tok)
    all_words = list(set(src_cntr + trg_cntr))

    vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + all_words

    w2id = {word: index for index, word in enumerate(vocab)}

    # Train Dataset
    source_data = make_dataset(source_path, w2id, args.tok)
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < args.max_seq_length
                  and 0 < len(t) < args.max_seq_length]

    # Display corpus statistics
    print("Vocab: {}".format(len(vocab)))
    print('Original training data size: %d' % len(source_data))
    print('Filtered training data size: %d' % len(train_data))

    # Valid Dataset
    source_path = os.path.join(args.input, args.source_valid)
    source_data = make_dataset(source_path, w2id, args.tok)
    target_path = os.path.join(args.input, args.target_valid)
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    valid_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) and 0 < len(t)]

    # Test Dataset
    source_path = os.path.join(args.input, args.source_test)
    source_data = make_dataset(source_path, w2id, args.tok)
    target_path = os.path.realpath(os.path.join(args.input, args.target_test))
    target_data = make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]

    id2w = {i: w for w, i in w2id.items()}

    # Save the dataset to numpy files
    np.save(os.path.join(args.input, args.save_data + '.train.npy'),
            train_data)
    np.save(os.path.join(args.input, args.save_data + '.valid.npy'),
            valid_data)
    np.save(os.path.join(args.input, args.save_data + '.test.npy'),
            test_data)

    # Save the vocab in json
    with open(os.path.join(args.input,
                           args.save_data + '.vocab.pickle'), 'wb') as f:
        pickle.dump(id2w,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)