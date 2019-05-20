# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import io
import itertools
import numpy as np
import random
from time import time
import torch
from torch import nn
import pickle
import shutil
import math
import logging
from torch.autograd import Variable

import evaluator
from models import MultiTaskNMT, Transformer, Shaped, LangShare
from exp_moving_avg import ExponentialMovingAverage
import optimizer as optim
from yogi import Yogi
from torchtext import data
import utils
from config import get_train_args
from fp16_utils import FP16_Optimizer, FP16_Module
from data_parallel import data_parallel as dp


def init_weights(m):
    if type(m) == nn.Linear:
        input_dim = m.weight.size(1)
        # LeCun Initialization
        m.weight.data.uniform_(-math.sqrt(3.0 / input_dim),
                                math.sqrt(3.0 / input_dim))
        # My custom initialization
        # m.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Xavier Initialization
        # output_dim = m.weight.size(0)
        # m.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                         math.sqrt(6.0 / (input_dim + output_dim)))

        if m.bias is not None:
            m.bias.data.fill_(0.)


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def save_checkpoint(state, is_best, model_path_, best_model_path_):
    torch.save(state, model_path_)
    if is_best:
        shutil.copyfile(model_path_, best_model_path_)


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[0]) + 2)
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def save_output(hypotheses, vocab, outf):
    # Save the Hypothesis to output file
    with io.open(outf, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def report_func(epoch, batch, num_batches, start_time, report_stats,
                report_every):
    """
    This is the user-defined batch-level training progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = utils.Statistics()
    return report_stats


# Have to unwrap DDP & FP16, if using.
def unwrap(module, model_name='Transformer'):
    if isinstance(module, eval(model_name)):
        return module
    return unwrap(module.module, model_name)


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_decode_len=50,
                 beam_size=1, alpha=0.6, max_sent=None):
        self.model = unwrap(model)
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_decode_length = max_decode_len
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_sent = max_sent

    def __call__(self, logger):
        self.model.eval()
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend(t.tolist() for t in targets)
            x_block = utils.source_pad_concat_convert(sources,
                                                      device=None)
            x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE),
                               requires_grad=False)
            ys = self.model.translate(x_block,
                                      self.max_decode_length,
                                      beam=self.beam_size,
                                      alpha=self.alpha)
            hypotheses.extend(ys)
            if self.max_sent is not None and \
                    ((i + 1) > self.max_sent):
                break

            # Log Progress
            if self.max_sent is not None:
                den = self.max_sent
            else:
                den = len(self.test_data)
            print("> Completed: [ %d / %d ]" % (i, den), end='\r')

        bleu = evaluator.BLEUEvaluator().evaluate(references, hypotheses)
        logger.info('BLEU: {}'.format(bleu.score_str()))
        logger.info('')
        return bleu.bleu, hypotheses


def main():
    best_score = 0
    args = get_train_args()
    logger = get_logger(args.log_path)
    logger.info(json.dumps(args.__dict__, indent=4))

    # Set seed value
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # Reading the int indexed text dataset
    train_data = np.load(os.path.join(args.input, args.data + ".train.npy"), allow_pickle=True)
    train_data = train_data.tolist()
    dev_data = np.load(os.path.join(args.input, args.data + ".valid.npy"), allow_pickle=True)
    dev_data = dev_data.tolist()
    test_data = np.load(os.path.join(args.input, args.data + ".test.npy"), allow_pickle=True)
    test_data = test_data.tolist()

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'), 'rb') as f:
        id2w = pickle.load(f)

    args.id2w = id2w
    args.n_vocab = len(id2w)

    # Define Model
    model = eval(args.model)(args)
    model.apply(init_weights)

    tally_parameters(model)
    if args.gpu >= 0:
        model.cuda(args.gpu)
        # model = data_parallel.DataParallel(model, device_ids=[0, 1]).cuda(0)
    logger.info(model)

    if args.optimizer == 'Noam':
        optimizer = optim.TransformerAdamTrainer(model, args)
    elif args.optimizer == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params,
                                     lr=args.learning_rate,
                                     betas=(args.optimizer_adam_beta1,
                                            args.optimizer_adam_beta2),
                                     eps=args.optimizer_adam_epsilon)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='max',
                                                               factor=0.7,
                                                               patience=7,
                                                               verbose=True)
    elif args.optimizer == 'Yogi':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Yogi(params,
                         lr=args.learning_rate,
                         betas=(args.optimizer_adam_beta1,
                                args.optimizer_adam_beta2),
                         eps=args.optimizer_adam_epsilon)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='max',
                                                               factor=0.7,
                                                               patience=7,
                                                               verbose=True)

    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16},
                                   verbose=False)
        
    ema = ExponentialMovingAverage(decay=args.ema_decay)
    ema.register(model.state_dict())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.model_file):
            logger.info("=> loading checkpoint '{}'".format(args.model_file))
            checkpoint = torch.load(args.model_file)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".
                  format(args.model_file, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.model_file))

    src_data, trg_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_trg_words = len(list(itertools.chain.from_iterable(trg_data)))
    iter_per_epoch = (total_src_words + total_trg_words) // (2 * args.wbatchsize)
    logger.info('Approximate number of iter/epoch = {}'.format(iter_per_epoch))
    time_s = time()

    global_steps = 0
    num_grad_steps = 0
    if args.grad_norm_for_yogi and args.optimizer == 'Yogi':
        args.start_epoch = -1
        l2_norm = 0.0
        parameters = list(filter(lambda p: p.requires_grad is True, model.parameters()))
        n_params = sum([p.nelement() for p in parameters])

    for epoch in range(args.start_epoch, args.epoch):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data,
                                        args.wbatchsize,
                                        key=lambda x: (len(x[0]), len(x[1])),
                                        batch_size_fn=batch_size_fn,
                                        random_shuffler=
                                        data.iterator.RandomShuffler())
        report_stats = utils.Statistics()
        train_stats = utils.Statistics()
        if args.debug:
            grad_norm = 0.
        for num_steps, train_batch in enumerate(train_iter):
            global_steps += 1
            model.train()
            if args.grad_accumulator_count == 1:
                optimizer.zero_grad()
            elif num_grad_steps % args.grad_accumulator_count == 0:
                optimizer.zero_grad()
            src_iter = list(zip(*train_batch))[0]
            src_words = len(list(itertools.chain.from_iterable(src_iter)))
            report_stats.n_src_words += src_words
            train_stats.n_src_words += src_words
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            loss, stat = model(*in_arrays)
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if len(args.multi_gpu) > 1:
                loss_tuple, stat_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                n_total = sum([obj.n_words for obj in stat_tuple])
                n_correct = sum([obj.n_correct for obj in stat_tuple])
                loss = 0
                for l_, s_ in zip(loss_tuple, stat_tuple):
                    loss += l_ * s_.n_words
                loss /= n_total
                stat = utils.Statistics(loss=loss.data.cpu() * n_total,
                                         n_correct=n_correct,
                                         n_words=n_total)
            else:
                loss, stat = model(*in_arrays)

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if epoch == -1 and args.grad_norm_for_yogi and args.optimizer == 'Yogi':
                l2_norm += (utils.grad_norm(model.parameters()) ** 2) / n_params
                continue
            num_grad_steps += 1
            if args.debug:
                norm = utils.grad_norm(model.parameters())
                grad_norm += norm
                if global_steps % args.report_every == 0:
                    logger.info("> Gradient Norm: %1.4f" % (grad_norm / (num_steps + 1)))
            if args.grad_accumulator_count == 1:
                optimizer.step()
                ema.apply(model.state_dict(keep_vars=True))
            elif num_grad_steps % args.grad_accumulator_count == 0:
                optimizer.step()
                ema.apply(model.state_dict(keep_vars=True))
                num_grad_steps = 0
            report_stats.update(stat)
            train_stats.update(stat)
            report_stats = report_func(epoch, num_steps, iter_per_epoch,
                                       time_s, report_stats, args.report_every)

            valid_stats = utils.Statistics()
            if global_steps % args.eval_steps == 0:
                dev_iter = data.iterator.pool(dev_data,
                                              args.wbatchsize,
                                              key=lambda x: (len(x[0]), len(x[1])),
                                              batch_size_fn=batch_size_fn,
                                              random_shuffler=data.iterator.
                                              RandomShuffler())

                for dev_batch in dev_iter:
                    model.eval()
                    in_arrays = utils.seq2seq_pad_concat_convert(dev_batch, -1)
                    if len(args.multi_gpu) > 1:
                        _, stat_tuple = zip(*dp(model, in_arrays, device_ids=args.multi_gpu))
                        n_total = sum([obj.n_words for obj in stat_tuple])
                        n_correct = sum([obj.n_correct for obj in stat_tuple])
                        dev_loss = sum([obj.loss for obj in stat_tuple])
                        stat = utils.Statistics(loss=dev_loss,
                                                n_correct=n_correct,
                                                n_words=n_total)
                    else:
                        _, stat = model(*in_arrays)
                    valid_stats.update(stat)

                logger.info('Train perplexity: %g' % train_stats.ppl())
                logger.info('Train accuracy: %g' % train_stats.accuracy())

                logger.info('Validation perplexity: %g' % valid_stats.ppl())
                logger.info('Validation accuracy: %g' % valid_stats.accuracy())

                if args.metric == "accuracy":
                    score = valid_stats.accuracy()
                elif args.metric == "bleu":
                    score, _ = CalculateBleu(model,
                                             dev_data,
                                             'Dev Bleu',
                                             batch=args.batchsize // 4,
                                             beam_size=args.beam_size,
                                             alpha=args.alpha,
                                             max_sent=args.max_sent_eval)(logger)

                # Threshold Global Steps to save the model
                if global_steps > 8000:
                    is_best = score > best_score
                    best_score = max(score, best_score)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'state_dict_ema': ema.shadow_variable_dict,
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict(),
                        'opts': args,
                    },  is_best,
                        args.model_file + '.{}'.format(epoch + 1),
                        args.best_model_file)

                if args.optimizer == 'Adam' or args.optimizer == 'Yogi':
                    scheduler.step(score)

        if epoch == -1 and args.grad_norm_for_yogi and args.optimizer == 'Yogi':
            optimizer.v_init = l2_norm / (num_steps + 1)
            logger.info("Initializing Yogi Optimizer (v_init = {})".format(optimizer.v_init))

    # BLEU score on Dev and Test Data
    checkpoint = torch.load(args.best_model_file)
    logger.info("=> loaded checkpoint '{}' (epoch {}, best score {})".
          format(args.best_model_file,
                 checkpoint['epoch'],
                 checkpoint['best_score']))
    model.load_state_dict(checkpoint['state_dict'])

    logger.info('Dev Set BLEU Score')
    _, dev_hyp = CalculateBleu(model,
                               dev_data,
                               'Dev Bleu',
                               batch=args.batchsize // 4,
                               beam_size=args.beam_size,
                               alpha=args.alpha,
                               max_decode_len=args.max_decode_len)(logger)
    save_output(dev_hyp, id2w, args.dev_hyp)

    logger.info('Test Set BLEU Score')
    _, test_hyp = CalculateBleu(model,
                                test_data,
                                'Test Bleu',
                                batch=args.batchsize // 4,
                                beam_size=args.beam_size,
                                alpha=args.alpha,
                                max_decode_len=args.max_decode_len)(logger)
    save_output(test_hyp, id2w, args.test_hyp)

    # Loading EMA state dict
    model.load_state_dict(checkpoint['state_dict_ema'])
    logger.info('Dev Set BLEU Score')
    _, dev_hyp = CalculateBleu(model,
                               dev_data,
                               'Dev Bleu',
                               batch=args.batchsize // 4,
                               beam_size=args.beam_size,
                               alpha=args.alpha,
                               max_decode_len=args.max_decode_len)(logger)
    save_output(dev_hyp, id2w, args.dev_hyp + '.ema')

    logger.info('Test Set BLEU Score')
    _, test_hyp = CalculateBleu(model,
                                test_data,
                                'Test Bleu',
                                batch=args.batchsize // 4,
                                beam_size=args.beam_size,
                                alpha=args.alpha,
                                max_decode_len=args.max_decode_len)(logger)
    save_output(test_hyp, id2w, args.test_hyp + '.ema')


if __name__ == '__main__':
    main()
