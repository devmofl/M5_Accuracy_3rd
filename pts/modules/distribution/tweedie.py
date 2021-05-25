import torch
import numpy as np
from torch.distributions.distribution import Distribution

def est_lambda(mu, p):
    return mu ** (2 - p) / (2 - p)

def est_alpha(p):
    return (2 - p) / (p - 1)    

def est_beta(mu, p):
    return mu ** (1 - p) / (p - 1)


class Tweedie(Distribution):
    r"""
    Creates a Tweedie distribution, i.e. distribution
    
    Args:    
        log_mu (Tensor): log(mean)
        rho (Tensor): tweedie_variance_power (1 ~ 2)
    """

    def __init__(self, log_mu, rho, validate_args=None):
        self.log_mu = log_mu
        self.rho = rho

        batch_shape = log_mu.size()
        super(Tweedie, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return torch.exp(self.log_mu)

    @property
    def variance(self):
        return torch.ones_line(self.log_mu) #TODO need to be assigned

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        mu = self.mean
        p = self.rho
        phi = 1 #TODO

        
        rate = est_lambda(mu, p) / phi     #rate for poisson
        alpha = est_alpha(p)             #alpha for Gamma distribution
        beta = est_beta(mu, p) / phi     #beta for Gamma distribution

        N = torch.poisson(rate)

        gamma = torch.distributions.gamma.Gamma(N*alpha, beta)
        samples = gamma.sample()
        samples[N==0] = 0

        return samples

    def log_prob(self, y_true):
        rho = self.rho
        y_pred = self.log_mu

        a = y_true * torch.exp((1 - rho) * y_pred) / (1 - rho)
        b = torch.exp((2 - rho) * y_pred) / (2 - rho)

        return a - b