import torch
from torch.distributions.distribution import Distribution

class ConstantDistribution(Distribution):
    r"""
    Creates a constant distribution, i.e. Var(x) = 0
    
    Args:    
        loss_type: L1 or L2
        mu (Tensor): mean
    """
    def __init__(self, loss_type, mu, validate_args=None):
        self.loss_type = loss_type
        self.mu = mu

        batch_shape = mu.size()
        super(ConstantDistribution, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return torch.zeros_like(self.mu)

    def sample(self, sample_shape=torch.Size()):
        return torch.ones_like(self.mu) * self.mu

    def log_prob(self, y_true):
        mu = self.mu

        if self.loss_type == "L1":
            loss = torch.abs(y_true - mu)
        elif self.loss_type == "L2":
            loss = (y_true - mu)**2
        else:
            raise NotImplementedError

        return -loss    # loss == negative log_prob