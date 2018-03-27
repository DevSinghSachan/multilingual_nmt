import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import utils
import preprocess


def where(cond, x_1, x_2):
    """
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/9
    :param cond:
    :param x_1:
    :param x_2:
    :return:
    """
    cond = cond.type_as(x_1)
    return (cond * x_1) + ((1 - cond) * x_2)


class PolynomialNormalization(object):
    """Dividing by the length (raised to some power (default 0.6))"""

    def __init__(self, alpha=0.6, apply_during_search=True):
        self.alpha = alpha
        self.apply_during_search = apply_during_search

    def lp(self, len):
        return pow(5 + len, self.alpha) / pow(5 + 1, self.alpha)

    def normalize_completed(self, completed_hyps, src_length=None):
        if not self.apply_during_search:
            for hyp in completed_hyps:
                hyp.score /= pow(len(hyp.id_list), self.m)

    def normalize_partial(self, score_so_far, score_to_add, new_len):
        if self.apply_during_search:
            return (score_so_far * self.lp(new_len - 1) + score_to_add) / self.lp(new_len)
        else:
            return score_so_far + score_to_add


def update_beam_state(outs, total_score, topk, topk_score, eos_id, alpha, x_block, z_blocks):
    full = outs.size()[0]
    prev_full, k = topk.size()
    batch = full // k
    prev_k = prev_full // batch
    assert (prev_k in [1, k])

    if total_score is None:
        total_score = topk_score
    else:
        is_end = torch.max(outs == eos_id, dim=1)[0]
        is_end = is_end.view(-1, 1).expand_as(topk_score)
        bias = torch.zeros_like(topk_score).type(utils.FLOAT_TYPE)
        bias[:, 1:] = -10000.  # remove ended cands except for a consequence

        obj = PolynomialNormalization(alpha=alpha)
        normalized_total_score = obj.normalize_partial(total_score[:, None],
                                                       topk_score,
                                                       outs.size()[1])

        # Use torch.where in v0.4
        total_score = where(Variable(is_end,
                                     requires_grad=False),
                            Variable(total_score[:, None] + bias,
                                     requires_grad=False),
                            Variable(normalized_total_score,
                                     requires_grad=False))

        total_score = total_score.data
        assert (torch.max(total_score) < 0.)

        # Use torch.where in v0.4
        topk = where(Variable(is_end,
                              requires_grad=False),
                     Variable(torch.LongTensor([eos_id]).type(utils.LONG_TYPE),
                              requires_grad=False),
                     Variable(topk,
                              requires_grad=False))  # this is not required
        topk = topk.data

    total_score = total_score.view((prev_full // prev_k, prev_k * k))
    total_topk_score, argtopk = torch.topk(total_score, k)

    assert (argtopk.size() == (prev_full // prev_k, k))
    assert (total_topk_score.size() == (prev_full // prev_k, k))

    total_topk = topk.take(argtopk + (torch.arange(prev_full // prev_k)[:, None] * prev_k * k).type(utils.LONG_TYPE))
    total_topk = total_topk.view((full,))
    total_topk_score = total_topk_score.view((full,))
    argtopk = argtopk / k + (torch.arange(prev_full // prev_k)[:, None] * prev_k).type(utils.LONG_TYPE)
    argtopk = argtopk.view((full,))

    xss = torch.split(x_block, 1)
    x_block = torch.cat([xss[i] for i in argtopk])

    zss = torch.split(z_blocks, 1)
    z_blocks = torch.cat([zss[i] for i in argtopk])

    outs = torch.split(outs, 1)
    outs = torch.cat([outs[i] for i in argtopk])
    outs = torch.cat([outs, total_topk[:, None]], dim=1).type(utils.LONG_TYPE)

    return outs, total_topk_score, x_block, z_blocks


def finish_beam(outs, total_score, batchsize, eos_id):
    k = outs.shape[0] // batchsize
    result_batch = collections.defaultdict(lambda: {'outs': [], 'score': -1e8})
    for i in range(batchsize):
        for j in range(k):
            score = total_score[i * k + j]
            if result_batch[i]['score'] < score:
                out = outs[i * k + j].tolist()
                if eos_id in out:
                    out = out[:out.index(eos_id)]
                result_batch[i] = {'outs': out, 'score': score}

    result_batch = [result for i, result in sorted(result_batch.items(), key=lambda x: x[0])]

    id_list, score_list = [], []
    for item in result_batch:
        id_list.append(item['outs'])
        score_list.append(item['score'])
    return id_list, score_list


class BeamSearch(object):
    def __init__(self, beam_size=5, max_len=50, alpha=0.6):
        self.max_decode_length = max_len
        self.k = beam_size
        self.alpha = alpha

    def generate_output(self, model, x_block):
        # x_block = utils.source_pad_concat_convert(x_block, device=None)
        batchsize, x_length = x_block.shape
        # self.max_decode_length = x_length + 50

        x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE), requires_grad=False)
        bos_array = np.array([[preprocess.Vocab_Pad.BOS]] * batchsize, 'i')
        y_block = Variable(torch.LongTensor(bos_array).type(utils.LONG_TYPE), requires_grad=False)

        outs = torch.LongTensor([[preprocess.Vocab_Pad.BOS]] * batchsize * self.k).type(utils.LONG_TYPE)
        total_score, z_blocks = None, None

        for i in range(self.max_decode_length):
            log_prob_tail, z_blocks = model(x_block,
                                            y_block,
                                            y_out_block=None,
                                            get_prediction=True,
                                            z_blocks=z_blocks)
            topk_score, topk = torch.topk(F.log_softmax(log_prob_tail, dim=1).data, self.k)
            assert (torch.max(topk_score) <= 0.)

            outs, total_score, x_block, z_blocks = update_beam_state(outs,
                                                                     total_score,
                                                                     topk,
                                                                     topk_score,
                                                                     preprocess.Vocab_Pad.EOS,
                                                                     self.alpha,
                                                                     x_block.data,
                                                                     z_blocks.data)
            assert (torch.max(total_score < 0.)), i
            y_block, x_block, z_blocks = Variable(outs), Variable(x_block), Variable(z_blocks)

            if torch.max(outs == preprocess.Vocab_Pad.EOS, 1)[0].sum() == outs.shape[0]:
                break  # all cands meet eos, end the loop
        result = finish_beam(outs[:, 1:], total_score, batchsize, preprocess.Vocab_Pad.EOS)
        return result


class GreedySearch(object):
    def __init__(self, max_len=50):
        self.max_decode_length = max_len

    def generate_output(self, model, x_block):
        # x_block = utils.source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        # self.max_decode_length = x_length + 50
        # bos
        y_block = np.full((batch, 1), preprocess.Vocab_Pad.BOS,
                          dtype=x_block.dtype)
        eos_flags = np.zeros((batch,), dtype=x_block.dtype)

        x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE),
                           requires_grad=False)
        y_block = Variable(torch.LongTensor(y_block).type(utils.LONG_TYPE),
                           requires_grad=False)

        result = []
        z_blocks = None
        for i in range(self.max_decode_length):
            log_prob_tail, z_blocks = model(x_block,
                                            y_block,
                                            y_out_block=None,
                                            get_prediction=True,
                                            z_blocks=z_blocks)
            _, ys = torch.max(log_prob_tail, dim=1)
            y_block = torch.cat([y_block.detach(), ys[:, None]], dim=1)
            ys = ys.data.cpu().numpy()
            result.append(ys)
            eos_flags += (ys == preprocess.Vocab_Pad.EOS)
            if np.all(eos_flags):
                break

        result = np.stack(result).T
        # Remove EOS tags
        outs = []
        for y in result:
            inds = np.argwhere(y == preprocess.Vocab_Pad.EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y.tolist())
        return outs
