import torch
def simulate_bernoulli_gamma(p, shape, rate, localize = 1):

    bernoulli_samples = torch.bernoulli(p)
    gamma_samples = torch.distributions.Gamma(shape, rate).sample() + localize
    
    return bernoulli_samples * gamma_samples