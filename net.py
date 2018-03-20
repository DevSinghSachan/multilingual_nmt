# encoding: utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import scipy.stats as stats

import utils
import search_strategy
import preprocess
from expert_utils import PadRemover

cudnn.benchmark = True


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):
    """Outputs random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean and standard deviation,
    except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    API from: https://www.tensorflow.org/api_docs/python/tf/truncated_normal"""
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev, (upper - mean) / stddev, loc=mean, scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a truncated normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters using Truncated Normal Initializer (default in Tensorflow)
        """
        # Initialize the embedding parameters (Default)
        # This works well too
        # self.embed_word.weight.data.uniform_(-3. / self.num_embeddings,
        #                                      3. / self.num_embeddings)

        self.weight.data = truncated_normal(shape=(self.num_embeddings,
                                                   self.embedding_dim),
                                            stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class TiedLinear(object):
    def __init__(self, param):
        self.weight = param

    def __call__(self, h):
        return F.linear(h, self.weight)


class LayerNorm(nn.Module):
    """Layer normalization module.
    Code adapted from OpenNMT-py open-source toolkit on 08/01/2018:
    URL: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py#L24"""

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) + self.b_2.expand_as(ln_out)
        return ln_out


def sentence_block_embed(embed, x):
    """Computes sentence-level embedding representation from word-ids.

    :param embed: nn.Embedding() Module
    :param x: Tensor of batched word-ids
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    """
    batch, length = x.shape
    _, units = embed.weight.size()
    e = embed(x).transpose(1, 2).contiguous()
    assert (e.size() == (batch, units, length))
    return e


def seq_func(func, x, reconstruct_shape=True, pad_remover=None):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length) or (batchsize x sentence_length, dimension)
    """
    batch, units, length = x.shape
    e = torch.transpose(x, 1, 2).contiguous().view(batch * length, units)
    if pad_remover:
        e = pad_remover.remove(e)
    e = func(e)
    if pad_remover:
        e = pad_remover.restore(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = torch.transpose(e.view((batch, length, out_units)), 1, 2).contiguous()
    assert (e.shape == (batch, out_units, length))
    return e


class LayerNormSent(LayerNorm):
    """Position-wise layer-normalization layer for array of shape (batchsize, dimension, sentence_length)."""

    def __init__(self, n_units, eps=1e-3):
        super(LayerNormSent, self).__init__(n_units, eps=eps)

    def forward(self, x):
        y = seq_func(super(LayerNormSent, self).forward, x)
        return y


class LinearSent(nn.Module):
    """Position-wise Linear Layer for sentence block. array of shape (batchsize, dimension, sentence_length)."""

    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        self.L.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Using Xavier Initialization
        # self.L.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                             math.sqrt(6.0 / (input_dim + output_dim)))
        # He Initialization
        # self.L.weight.data.uniform_(-math.sqrt(6.0 / input_dim), math.sqrt(6.0 / input_dim))

        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, x, pad_remover=None):
        # output = self.L.weight.matmul(x)
        # if self.L.bias is not None:
        #    output += self.L.bias.unsqueeze(-1)
        output = seq_func(self.L, x, pad_remover=pad_remover)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer for Sentence Blocks. For computational efficiency,
    dot-product to calculate query-key scores is performed in all the heads together.
    Positional Attention is introduced in "Non-Autoregressive Neural Machine Translation"
    (https://arxiv.org/abs/1711.02281)
    """

    def __init__(self, n_units, multi_heads=8, attention_dropout=0.1, pos_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = LinearSent(n_units, n_units, bias=False)
        self.W_K = LinearSent(n_units, n_units, bias=False)
        self.W_V = LinearSent(n_units, n_units, bias=False)
        self.finishing_linear_layer = LinearSent(n_units, n_units, bias=False)
        self.h = multi_heads
        self.pos_attn = pos_attn
        self.scale_score = 1. / (n_units // multi_heads) ** 0.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)

        if not self.pos_attn:
            if z is None:
                K, V = self.W_K(x), self.W_V(x)
            else:
                K, V = self.W_K(z), self.W_V(z)
        else:
            K, V = self.W_K(x), self.W_V(z)

        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate attention scores with mask for zero-padded areas
        # Perform multi-head attention using pseudo batching all together at once for efficiency
        Q = torch.cat(torch.chunk(Q, h, dim=1), dim=0)
        K = torch.cat(torch.chunk(K, h, dim=1), dim=0)
        V = torch.cat(torch.chunk(V, h, dim=1), dim=0)

        assert (Q.shape == (batch * h, n_units // h, n_querys))
        assert (K.shape == (batch * h, n_units // h, n_keys))
        assert (V.shape == (batch * h, n_units // h, n_keys))

        mask = torch.cat([mask] * h, dim=0)
        Q = Q.transpose(1, 2).contiguous() * self.scale_score
        batch_A = torch.bmm(Q, K)

        batch_A = batch_A.masked_fill(1. - mask, -np.inf)
        batch_A = F.softmax(batch_A, dim=2)

        # Replaces 'NaN' with zeros and other values with the original ones
        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # Attention Dropout
        batch_A = self.dropout(batch_A)

        # Calculate Weighted Sum
        V = V.transpose(1, 2).contiguous()
        C = torch.transpose(torch.bmm(batch_A, V), 1, 2).contiguous()
        assert (C.shape == (batch * h, n_units // h, n_querys))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=1)
        assert (C.shape == (batch, n_units, n_querys))

        # Final linear layer
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units, n_hidden, relu_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = LinearSent(n_units, n_hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(relu_dropout)
        self.W_2 = LinearSent(n_hidden, n_units)

    def forward(self, e, pad_remover=None):
        e = self.W_1(e, pad_remover=pad_remover)
        e = self.dropout(self.act(e))
        e = self.W_2(e, pad_remover=pad_remover)
        return e


class EncoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8, layer_prepostprocess_dropout=0.1,
                 n_hidden=2048, attention_dropout=0.1, relu_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.ln_1 = LayerNormSent(n_units, eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units, multi_heads, attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_2 = LayerNormSent(n_units, eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units, n_hidden, relu_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, xx_mask, pad_remover=None):
        e = self.ln_1(e)
        sub = self.self_attention(e, mask=xx_mask)
        e = e + self.dropout1(sub)

        e = self.ln_2(e)
        sub = self.feed_forward(e, pad_remover=pad_remover)
        e = e + self.dropout2(sub)
        return e


class DecoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8, layer_prepostprocess_dropout=0.1,
                 pos_attention=False, n_hidden=2048, attention_dropout=0.1, relu_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.pos_attention = pos_attention

        self.ln_1 = LayerNormSent(n_units, eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units, multi_heads, attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)

        if pos_attention:
            position_encoding_block = Transformer.initialize_position_encoding(500, n_units)
            self.position_encoding_block = nn.Parameter(torch.FloatTensor(position_encoding_block), requires_grad=False)
            self.register_parameter("Position Encoding Block", self.position_encoding_block)

            self.ln_pos = LayerNormSent(n_units, eps=1e-3)
            self.pos_attention = MultiHeadAttention(n_units, multi_heads, attention_dropout, pos_attn=True)
            self.dropout_pos = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_2 = LayerNormSent(n_units, eps=1e-3)
        self.source_attention = MultiHeadAttention(n_units, multi_heads, attention_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_3 = LayerNormSent(n_units, eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units, n_hidden, relu_dropout)
        self.dropout3 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, s, xy_mask, yy_mask, pad_remover):
        batch, units, length = e.shape

        e = self.ln_1(e)
        sub = self.self_attention(e, mask=yy_mask)
        e = e + self.dropout1(sub)

        if self.pos_attention:
            e = self.ln_pos(e)
            p = self.position_encoding_block[:, :, :length]
            p = p.expand(batch, units, length)
            sub = self.pos_attention(p, e, mask=yy_mask)
            e = e + self.dropout_pos(sub)

        e = self.ln_2(e)
        sub = self.source_attention(e, s, mask=xy_mask)
        e = e + self.dropout2(sub)

        e = self.ln_3(e)
        sub = self.feed_forward(e, pad_remover=pad_remover)
        e = e + self.dropout3(sub)
        return e


class Encoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8, layer_prepostprocess_dropout=0.1,
                 n_hidden=2048, attention_dropout=0.1, relu_dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = EncoderLayer(n_units, multi_heads, layer_prepostprocess_dropout,
                                 n_hidden, attention_dropout, relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, xx_mask, pad_remover):
        for layer in self.layers:
            e = layer(e, xx_mask, pad_remover)
        e = self.ln(e)
        return e


class Decoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8, layer_prepostprocess_dropout=0.1,
                 pos_attention=False, n_hidden=2048, attention_dropout=0.1,
                 relu_dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = DecoderLayer(n_units, multi_heads, layer_prepostprocess_dropout,
                                 pos_attention, n_hidden, attention_dropout, relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, source, xy_mask, yy_mask, pad_remover):
        for layer in self.layers:
            e = layer(e, source, xy_mask, yy_mask, pad_remover)
        e = self.ln(e)
        return e


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embed_word = ScaledEmbedding(config.n_vocab, config.n_units, padding_idx=0)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.n_hidden = config.n_units * 4
        self.encoder = Encoder(config.layers, config.n_units, config.multi_heads,
                               config.layer_prepostprocess_dropout, self.n_hidden,
                               config.attention_dropout, config.relu_dropout)
        self.decoder = Decoder(config.layers, config.n_units, config.multi_heads,
                               config.layer_prepostprocess_dropout, config.pos_attention,
                               self.n_hidden, config.attention_dropout, config.relu_dropout)

        if config.embed_position:
            self.embed_pos = nn.Embedding(config.max_length, config.n_units, padding_idx=0)

        if config.tied:
            self.affine = TiedLinear(self.embed_word.weight)
        else:
            self.affine = nn.Linear(config.n_units, config.n_vocab, bias=False)

        self.n_target_vocab = config.n_vocab
        self.dropout = config.dropout
        self.label_smoothing = config.label_smoothing
        self.scale_emb = config.n_units ** 0.5

        position_encoding_block = self.initialize_position_encoding(config.max_length,
                                                                    config.n_units)
        self.position_encoding_block = nn.Parameter(torch.FloatTensor(position_encoding_block),
                                                    requires_grad=False)
        self.register_parameter("Position Encoding Block", self.position_encoding_block)

    @staticmethod
    def initialize_position_encoding(length, emb_dim):
        channels = emb_dim
        position = np.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.reshape(signal, [1, length, channels])
        position_encoding_block = np.transpose(signal, (0, 2, 1))
        return position_encoding_block

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += self.position_encoding_block[:, :, :length]

        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(self.embed_pos,
                                              np.broadcast_to(np.arange(length).astype('i')[None, :],
                                                              block.shape))
        emb_block = self.embed_dropout(emb_block)
        return emb_block

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 1) * \
               (source_block[:, :, None] >= 1)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = np.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[None,]
        history_mask = np.broadcast_to(history_mask, (batch, length, length))
        history_mask = history_mask.astype(np.int32)
        history_mask = Variable(torch.ByteTensor(history_mask).type(utils.BYTE_TYPE),
                                requires_grad=False)
        return history_mask

    def output(self, h):
        return self.affine(h)

    def output_and_loss(self, h_block, t_block):
        batch, units, length = h_block.shape

        # Output (all together at once for efficiency)
        concat_logit_block = seq_func(self.affine, h_block, reconstruct_shape=False)
        rebatch, _ = concat_logit_block.shape

        # Make target
        concat_t_block = t_block.view(rebatch)
        ignore_mask = (concat_t_block >= 1).float()
        n_token = torch.sum(ignore_mask)
        normalizer = n_token

        if self.label_smoothing:
            log_prob = F.log_softmax(concat_logit_block, dim=1)
            broad_ignore_mask = ignore_mask[:, None].expand_as(concat_logit_block)
            pre_loss = ignore_mask * log_prob[np.arange(rebatch), concat_t_block]
            loss = -1. * torch.sum(pre_loss) / normalizer
        else:
            loss = F.cross_entropy(concat_logit_block, concat_t_block, ignore_index=0)

        n_correct, n_total = utils.accuracy(concat_logit_block, concat_t_block, ignore_index=0)
        stats = utils.Statistics(loss=utils.to_cpu(loss) * n_total, n_correct=utils.to_cpu(n_correct), n_words=n_total)

        if self.label_smoothing:
            pre_loss = (1 - self.label_smoothing) * loss
            ls_loss = -1. / self.n_target_vocab * broad_ignore_mask * log_prob
            ls_loss = torch.sum(ls_loss) / normalizer
            loss = pre_loss + (self.label_smoothing * ls_loss)

        return loss, stats

    def forward(self, x_block, y_in_block, y_out_block, get_prediction=False, z_blocks=None):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        if z_blocks is None:
            ex_block = self.make_input_embedding(self.embed_word, x_block)
            xx_mask = self.make_attention_mask(x_block, x_block)
            xpad_obj = PadRemover(x_block >= preprocess.Vocab_Pad.PAD)
            # Encode Sources
            z_blocks = self.encoder(ex_block, xx_mask, xpad_obj)
            # (batch, n_units, x_length)

        ey_block = self.make_input_embedding(self.embed_word, y_in_block)
        # Make Masks
        xy_mask = self.make_attention_mask(y_in_block, x_block)
        yy_mask = self.make_attention_mask(y_in_block, y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Create PadRemover objects
        ypad_obj = PadRemover(y_in_block >= preprocess.Vocab_Pad.PAD)

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask, ypad_obj)
        # (batch, n_units, y_length)

        if get_prediction:
            return self.output(h_block[:, :, -1]), z_blocks
        else:
            return self.output_and_loss(h_block, y_out_block)

    def translate(self, x_block, max_length=50, beam=5, alpha=0.6):
        if beam:
            obj = search_strategy.BeamSearch(beam_size=beam, max_len=max_length, alpha=alpha)
            id_list, score = obj.generate_output(self, x_block)
            return id_list

        x_block = utils.source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        y_block = np.full((batch, 1), preprocess.Vocab_Pad.BOS, dtype=x_block.dtype)  # bos
        eos_flags = np.zeros((batch,), dtype=x_block.dtype)

        x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE), requires_grad=False)
        y_block = Variable(torch.LongTensor(y_block).type(utils.LONG_TYPE), requires_grad=False)

        result = []
        z_blocks = None
        for i in range(max_length):
            log_prob_tail, z_blocks = self(x_block, y_block, y_out_block=None, get_prediction=True, z_blocks=z_blocks)
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
            outs.append(y)
        return outs
