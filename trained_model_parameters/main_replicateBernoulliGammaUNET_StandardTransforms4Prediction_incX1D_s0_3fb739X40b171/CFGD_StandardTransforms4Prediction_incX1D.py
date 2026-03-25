import torch


class YPrTransforms:
    """ Dummy transform"""
    def __init__(self, yy_pr, eps=1e-6):
        self.eps = eps

    def transform(self, y):           # [-1,1]
        return y

    def inverse(self, z):
        return z
    




class XPrTransforms:

    def __init__(self, xx_pr, eps=1e-8):
        self.eps = eps
        
        self.mu = xx_pr.mean(dim=(0, 2, 3), keepdim=True)
        self.sd = xx_pr.std(dim=(0, 2, 3), keepdim=True)

    def transform(self, x):
        z0 = (x - self.mu) / (self.sd + self.eps)
        return z0

    def inverse(self, z):
        x = z * (self.sd + self.eps) + self.mu
        return x
    


class XPrTransforms1D:

    def __init__(self, xx_pr, eps=1e-8):
        self.eps = eps
        
        self.mu = xx_pr.mean(dim=0, keepdim=True)
        self.sd = xx_pr.std(dim=0, keepdim=True)

    def transform(self, x):
        z0 = (x - self.mu) / (self.sd + self.eps)
        return z0

    def inverse(self, z):
        x = z * (self.sd + self.eps) + self.mu
        return x


