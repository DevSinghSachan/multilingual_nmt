import json
import numpy as np
import torch
import time
import math
import sys
from torch.autograd import Variable
# import convert
from chainer.dataset import convert


if torch.cuda.is_available():
    FLOAT_TYPE = torch.cuda.FloatTensor
    INT_TYPE = torch.cuda.IntTensor
    LONG_TYPE = torch.cuda.LongTensor
    BYTE_TYPE = torch.cuda.ByteTensor
else:
    FLOAT_TYPE = torch.FloatTensor
    INT_TYPE = torch.IntTensor
    LONG_TYPE = torch.LongTensor
    BYTE_TYPE = torch.ByteTensor


def to_cpu(x):
    try:
        y = x.data.cpu().tolist()[0]
    except:
        y = x.data.cpu().tolist()
    return y


class Accuracy(object):
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, y, t):
        if self.ignore_index is not None:
            mask = (t == self.ignore_index)
            ignore_cnt = torch.sum(mask.float())
            _, pred = torch.max(y, dim=1)
            pred = Variable(pred.view(t.shape))
            pred = pred.masked_fill(Variable(mask),
                                    self.ignore_index)
            count = torch.sum((pred.data == t).float()) - ignore_cnt
            total = torch.numel(t) - ignore_cnt

            if total == 0:
                return torch.FloatTensor([0.0])
            else:
                # return count / total
                return count, total
        else:
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            return torch.mean((pred == t).float())


def accuracy(y, t, ignore_index=None):
    return Accuracy(ignore_index=ignore_index)(y, t)


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=1, bos_id=3):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)
    x_block = convert.concat_examples(x_seqs, device, padding=0)
    y_block = convert.concat_examples(y_seqs, device, padding=0)

    # Add EOS
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)

    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    x_block = np.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)

    y_out_block = np.pad(y_block, ((0, 0), (0, 1)), 'constant', constant_values=0)

    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = np.pad(y_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)

    # Converting from numpy format to Torch Tensor
    x_block = Variable(torch.LongTensor(x_block).type(LONG_TYPE),
                       requires_grad=False)
    y_in_block = Variable(torch.LongTensor(y_in_block).type(LONG_TYPE),
                          requires_grad=False)
    y_out_block = Variable(torch.LongTensor(y_out_block).type(LONG_TYPE),
                           requires_grad=False)

    return x_block, y_in_block, y_out_block


def source_pad_concat_convert(x_seqs, device, eos_id=1, bos_id=3):
    x_block = convert.concat_examples(x_seqs, device, padding=0)

    # add eos
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    x_block = np.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)
    return x_block


class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {self._decode(k): v for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * elapsed time
    Code adapted from OpenNMT-py open-source toolkit on 10/01/2018:
    URL: https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        avg_loss = self.loss / self.n_words
        if (avg_loss < 100).all():
            return math.exp(avg_loss)
        else:
            return math.exp(100)
        # return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.
        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


def grad_norm(parameters, norm_type=2):
    r"""The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        norm_type (float or int): type of the used p-norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
