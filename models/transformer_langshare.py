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


def input_like(tensor, val=0):
    """
    Use clone() + fill_() to make sure that a tensor ends up on the right
    device at runtime.
    """
    return tensor.clone().fill_(val)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):
    """Outputs random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean
    and standard deviation, except that values whose magnitude is more
    than 2 standard deviations from the mean are dropped and re-picked.
    API from: https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev,
                        (upper - mean) / stddev,
                        loc=mean,
                        scale=stddev)
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
                                            stddev=1.0 / math.sqrt(
                                                self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def sentence_block_embed(embed, x):
    """Computes sentence-level embedding representation from word-ids.

    :param embed: nn.Embedding() Module
    :param x: Tensor of batched word-ids
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    """
    batch, length = x.shape
    _, units = embed.weight.size()
    e = embed(x)
    assert (e.size() == (batch, length, units))
    return e


def seq_func(func, x, reconstruct_shape=True, pad_remover=None):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be
    of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    or (batchsize x sentence_length, dimension)
    """
    batch, length, units = x.shape
    e = x.view(batch * length, units)
    if pad_remover:
        e = pad_remover.remove(e)
    e = func(e)
    if pad_remover:
        e = pad_remover.restore(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = e.view(batch, length, out_units)
    assert (e.shape == (batch, length, out_units))
    return e


class LayerNormSent(LayerNorm):
    """Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length)."""

    def __init__(self, n_units, eps=1e-3):
        super(LayerNormSent, self).__init__(n_units, eps=eps)

    def forward(self, x):
        y = seq_func(super(LayerNormSent, self).forward, x)
        return y


class LinearSent(nn.Module):
    """Position-wise Linear Layer for sentence block. array of shape
    (batchsize, dimension, sentence_length)."""

    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        # self.L.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Using Xavier Initialization
        # self.L.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                             math.sqrt(6.0 / (input_dim + output_dim)))
        # LeCun Initialization
        self.L.weight.data.uniform_(-math.sqrt(3.0 / input_dim),
                                    math.sqrt(3.0 / input_dim))

        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, x, pad_remover=None):
        output = seq_func(self.L, x, pad_remover=pad_remover)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer for Sentence Blocks.
    For computational efficiency, dot-product to calculate
    query-key scores is performed in all the heads together.
    Positional Attention is introduced in
    "Non-Autoregressive Neural Machine Translation"
    (https://arxiv.org/abs/1711.02281)
    """

    def __init__(self, n_units, multi_heads=8, attention_dropout=0.1,
                 pos_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(n_units,
                             n_units,
                             bias=False)
        self.W_K = nn.Linear(n_units,
                             n_units,
                             bias=False)
        self.W_V = nn.Linear(n_units,
                             n_units,
                             bias=False)
        self.finishing_linear_layer = nn.Linear(n_units,
                                                n_units,
                                                bias=False)
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

        batch, n_querys, n_units = Q.shape
        _, n_keys, _ = K.shape

        # Calculate attention scores with mask for zero-padded areas
        # Perform multi-head attention using pseudo batching all together
        # at once for efficiency
        Q = torch.cat(torch.chunk(Q, h, dim=2), dim=0)
        K = torch.cat(torch.chunk(K, h, dim=2), dim=0)
        V = torch.cat(torch.chunk(V, h, dim=2), dim=0)

        assert (Q.shape == (batch * h, n_querys, n_units // h))
        assert (K.shape == (batch * h, n_keys, n_units // h))
        assert (V.shape == (batch * h, n_keys, n_units // h))

        mask = torch.cat([mask] * h, dim=0)
        Q.mul_(self.scale_score)
        batch_A = torch.bmm(Q, K.transpose(1, 2).contiguous())

        # batch_A = batch_A.masked_fill(1. - mask, -np.inf) # Works in v0.4
        batch_A = batch_A.masked_fill(mask == 0, -1e18)
        batch_A = F.softmax(batch_A, dim=2)

        # Replaces 'NaN' with zeros and other values with the original ones
        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # Attention Dropout
        batch_A = self.dropout(batch_A)

        # Calculate Weighted Sum
        C = torch.bmm(batch_A, V)
        assert (C.shape == (batch * h, n_querys, n_units // h))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=2)
        assert (C.shape == (batch, n_querys, n_units))

        # Final linear layer
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units, n_hidden, relu_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(n_units, n_hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(relu_dropout)
        self.W_2 = nn.Linear(n_hidden, n_units)

    def forward(self, e, pad_remover=None):
        e = self.dropout(self.act(self.W_1(e)))
        e = self.W_2(e)
        return e


class EncoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.ln_1 = LayerNorm(n_units,
                              eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units,
                                                 multi_heads,
                                                 attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)
        self.ln_2 = LayerNorm(n_units,
                              eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units,
                                             n_hidden,
                                             relu_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, xx_mask, pad_remover=None):
        sub = self.self_attention(self.ln_1(e),
                                  mask=xx_mask)
        e = e + self.dropout1(sub)

        sub = self.feed_forward(self.ln_2(e),
                                pad_remover=pad_remover)
        e = e + self.dropout2(sub)
        return e


class DecoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1,
                 pos_attention=False, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.pos_attention = pos_attention
        self.ln_1 = LayerNorm(n_units,
                              eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units,
                                                 multi_heads,
                                                 attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)

        if pos_attention:
            pos_enc_block = TransformerLangShare.initialize_position_encoding(500,
                                                                              n_units)
            self.pos_enc_block = nn.Parameter(torch.FloatTensor(pos_enc_block),
                                              requires_grad=False)
            self.register_parameter("Position Encoding Block",
                                    self.pos_enc_block)

            self.ln_pos = LayerNorm(n_units,
                                    eps=1e-3)
            self.pos_attention = MultiHeadAttention(n_units,
                                                    multi_heads,
                                                    attention_dropout,
                                                    pos_attn=True)
            self.dropout_pos = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_2 = LayerNorm(n_units,
                              eps=1e-3)
        self.source_attention = MultiHeadAttention(n_units,
                                                   multi_heads,
                                                   attention_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_3 = LayerNorm(n_units,
                              eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units,
                                             n_hidden,
                                             relu_dropout)
        self.dropout3 = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_4 = LayerNorm(n_units,
                              eps=1e-3)
        self.feed_forward_lang = FeedForwardLayer(n_units,
                                                  n_hidden,
                                                  relu_dropout)
        self.dropout4 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, s, xy_mask, yy_mask, pad_remover):
        batch, units, length = e.shape
        ##### New
        sub = self.feed_forward_lang(self.ln_4(e),
                                     pad_remover=pad_remover)
        e = e + self.dropout4(sub)
        #####

        sub = self.self_attention(self.ln_1(e),
                                  mask=yy_mask)
        e = e + self.dropout1(sub)
        if self.pos_attention:
            p = self.pos_enc_block[:, :length, :]
            p = p.expand(batch, length, units)
            sub = self.pos_attention(p,
                                     self.ln_pos(e),
                                     mask=yy_mask)
            e = e + self.dropout_pos(sub)
        sub = self.source_attention(self.ln_2(e),
                                    s,
                                    mask=xy_mask)
        e = e + self.dropout2(sub)
        sub = self.feed_forward(self.ln_3(e),
                                pad_remover=pad_remover)
        e = e + self.dropout3(sub)

        return e


class Encoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = EncoderLayer(n_units,
                                 multi_heads,
                                 layer_prepostprocess_dropout,
                                 n_hidden,
                                 attention_dropout,
                                 relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNorm(n_units,
                            eps=1e-3)

    def forward(self, e, xx_mask, pad_remover):
        for layer in self.layers:
            e = layer(e,
                      xx_mask,
                      pad_remover)
        e = self.ln(e)
        return e


class Decoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, pos_attention=False,
                 n_hidden=2048, attention_dropout=0.1,
                 relu_dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = DecoderLayer(n_units,
                                 multi_heads,
                                 layer_prepostprocess_dropout,
                                 pos_attention,
                                 n_hidden,
                                 attention_dropout,
                                 relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNorm(n_units,
                            eps=1e-3)

    def forward(self, e, source, xy_mask, yy_mask, pad_remover):
        for layer in self.layers:
            e = layer(e,
                      source,
                      xy_mask,
                      yy_mask,
                      pad_remover)
        e = self.ln(e)
        return e


class TransformerLangShare(nn.Module):
    def __init__(self, config):
        super(TransformerLangShare, self).__init__()
        self.scale_emb = config.n_units ** 0.5
        self.padding_idx = 0
        self.embed_word = ScaledEmbedding(config.n_vocab,
                                          config.n_units,
                                          padding_idx=self.padding_idx)
        pos_enc_block = self.initialize_position_encoding(config.max_length,
                                                          config.n_units)
        self.pos_enc_block = nn.Parameter(torch.FloatTensor(pos_enc_block),
                                          requires_grad=False)
        self.register_parameter("Position Encoding Block",
                                self.pos_enc_block)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.n_hidden = config.n_units * 4
        self.encoder = Encoder(config.layers,
                               config.n_units,
                               config.multi_heads,
                               config.layer_prepostprocess_dropout,
                               self.n_hidden,
                               config.attention_dropout,
                               config.relu_dropout)

        self.decoder = Decoder(config.layers,
                               config.n_units,
                               config.multi_heads,
                               config.layer_prepostprocess_dropout,
                               config.pos_attention,
                               self.n_hidden,
                               config.attention_dropout,
                               config.relu_dropout)
        self.use_pad_remover = config.use_pad_remover

        if config.embed_position:
            self.embed_pos = nn.Embedding(config.max_length,
                                          config.n_units,
                                          padding_idx=0)
        if config.tied:
            self.affine = self.tied_linear
            self.affine_bias = nn.Parameter(torch.Tensor(config.n_vocab))
            stdv = 1. / math.sqrt(config.n_units)
            self.affine_bias.data.uniform_(-stdv, stdv)
        else:
            self.affine = nn.Linear(config.n_units,
                                    config.n_vocab,
                                    bias=True)
        self.n_target_vocab = config.n_vocab
        self.dropout = config.dropout
        self.label_smoothing = config.label_smoothing
        assert (0.0 <= self.label_smoothing <= 1.0)
        if self.label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False,
                                          reduce=True)
            one_hot = torch.randn(1, config.n_vocab)
            one_hot.fill_(self.label_smoothing / (config.n_vocab - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(config.n_vocab)
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight,
                                        size_average=False)
        self.confidence = 1.0 - self.label_smoothing

    @staticmethod
    def initialize_position_encoding(length, emb_dim):
        channels = emb_dim
        position = np.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (
                    np.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * np.exp(
            np.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(
            inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)],
                                axis=1)
        signal = np.reshape(signal, [1, length, channels])
        return signal

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += self.pos_enc_block[:, :length, :]

        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(self.embed_pos,
                                              np.broadcast_to(
                                                  np.arange(length).astype('i')[
                                                  None, :],
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
        history_mask = np.broadcast_to(history_mask,
                                       (batch, length, length))
        history_mask = history_mask.astype(np.int32)
        history_mask = Variable(
            torch.ByteTensor(history_mask).type(utils.BYTE_TYPE),
            requires_grad=False)
        return history_mask

    def tied_linear(self, h):
        return F.linear(h, self.embed_word.weight, self.affine_bias)

    def output(self, h):
        return self.affine(h)

    def output_and_loss(self, h_block, t_block):
        batch, length, units = h_block.shape
        # shape : (batch * sequence_length, num_classes)
        logits_flat = seq_func(self.affine,
                               h_block,
                               reconstruct_shape=False)
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat,
                                       dim=-1)
        rebatch, _ = logits_flat.shape
        concat_t_block = t_block.view(rebatch)
        weights = (concat_t_block >= 1).float()
        n_correct, n_total = utils.accuracy(logits_flat.data,
                                            concat_t_block.data,
                                            ignore_index=0)
        if self.confidence < 1:
            tdata = concat_t_block.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            tmp_ = self.one_hot.repeat(concat_t_block.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0 and mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            concat_t_block = Variable(tmp_, requires_grad=False)
        loss = self.criterion(log_probs_flat,
                              concat_t_block)
        loss = loss.sum() / (weights.sum() + 1e-13)
        stats = utils.Statistics(loss=loss.data.cpu() * n_total,
                                 n_correct=n_correct,
                                 n_words=n_total)
        return loss, stats

    def forward(self, x_block, y_in_block, y_out_block, get_prediction=False,
                z_blocks=None):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        if z_blocks is None:
            ex_block = self.make_input_embedding(self.embed_word,
                                                 x_block)
            xx_mask = self.make_attention_mask(x_block,
                                               x_block)
            xpad_obj = None
            if self.use_pad_remover:
                xpad_obj = PadRemover(x_block >= preprocess.Vocab_Pad.PAD)
            # Encode Sources
            z_blocks = self.encoder(ex_block,
                                    xx_mask,
                                    xpad_obj)
            # (batch, n_units, x_length)

        ey_block = self.make_input_embedding(self.embed_word,
                                             y_in_block)
        # Make Masks
        xy_mask = self.make_attention_mask(y_in_block,
                                           x_block)
        yy_mask = self.make_attention_mask(y_in_block,
                                           y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Create PadRemover objects
        ypad_obj = None
        if self.use_pad_remover:
            ypad_obj = PadRemover(y_in_block >= preprocess.Vocab_Pad.PAD)

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block,
                               z_blocks,
                               xy_mask,
                               yy_mask,
                               ypad_obj)
        # (batch, n_units, y_length)

        if get_prediction:
            return self.output(h_block[:, -1, :]), z_blocks
        else:
            return self.output_and_loss(h_block,
                                        y_out_block)

    def translate(self, x_block, max_length=50, beam=5, alpha=0.6):
        if beam > 1:
            obj = search_strategy.BeamSearch(beam_size=beam,
                                             max_len=max_length,
                                             alpha=alpha)
            id_list, score = obj.generate_output(self,
                                                 x_block)
            return id_list
        else:
            obj = search_strategy.GreedySearch(max_len=max_length)
            id_list = obj.generate_output(self,
                                          x_block)
            return id_list
