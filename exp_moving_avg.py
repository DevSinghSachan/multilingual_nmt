

class ExponentialMovingAverage(object):
    """
    ExponentialMoving
    """
    def __init__(self, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow_variable_dict = {}
        self.requires_grad_set = set()

    def register(self, var_list):
        for name, param in var_list.items():
            self.shadow_variable_dict[name] = param.clone()

    def mark_require_grad(self, var_list):
        for name, param in var_list:
            if param.requires_grad:
                self.requires_grad_set.add(name)

    def apply(self, var_list):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in var_list.items():
            if name in self.requires_grad_set:
                assert name in self.shadow_variable_dict
                data = self.shadow_variable_dict[name]
                data -= (1 - decay) * (data - param.clone())
