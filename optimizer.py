from __future__ import division
import numpy as np
import torch


class TransformerAdamTrainer(object):
    """
    Proposed in the paper "Attention is all you need"
    (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    [Page 7, Eq. 3] In this the learning rate of Adam Optimizer is
    increased for the first warmup steps followed
    by a gradual decay. In the paper, warmup steps = 4000, beta2=0.98
    """
    def __init__(self, model, config):
        params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=config.learning_rate,
                                          betas=(config.optimizer_adam_beta1,
                                                 config.optimizer_adam_beta2),
                                          eps=config.optimizer_adam_epsilon)
        self.dim = config.n_units
        self.warmup_steps = config.warmup_steps
        self.learning_rate = config.learning_rate
        self.learning_rate_constant = config.learning_rate_constant
        self.steps = 0

    def step(self):
        self.steps += 1
        decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5),
                                               self.steps * (self.warmup_steps ** (-1.5))])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate_constant * decay
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, params):
        return self.optimizer.load_state_dict(params)

