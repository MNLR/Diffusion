import torch




class YPrTransforms:
    """
    Precip target transform:
      yy_pr -> log1p -> minmax -> [-1,1]
    Stats computed from training yy_pr.
    """

    def __init__(self, yy_pr, eps=1e-6):
        self.eps = eps

        yy_pr = torch.clamp(yy_pr, min=0.0)
        yy_sqrt = torch.sqrt(yy_pr)

        self.vmin = yy_sqrt.amin(dim=(0, 2, 3), keepdim=True)
        self.vmax = yy_sqrt.amax(dim=(0, 2, 3), keepdim=True)
        self.denom = (self.vmax - self.vmin).clamp(min=self.eps)

    def transform(self, y):
        y = torch.clamp(y, min=0.0)
        y_sqrt = torch.sqrt(y)

        u = (y_sqrt - self.vmin) / self.denom          # [0,1]
        z = 2.0 * u - 1.0                              # [-1,1]
        return z.detach().clone()

    def inverse(self, z):
        u = 0.5 * (z + 1.0)                            # [0,1]
        y_sqrt = u * self.denom + self.vmin
        y = y_sqrt ** 2                                # square the sqrt to get back to original scale
        return torch.clamp(y, min=0.0).detach().clone()
    




class XPrTransforms:
    """
    Predictors transform for xx_pr with channels:
      0: pr_base  -> sqrt + zscore
      1: p        -> zscore (raw)
      2: alpha    -> zscore (raw)
      3: beta     -> zscore (raw)
    Statistics computed from training xx_pr.
    """

    def __init__(self, xx_pr, eps=1e-6):
        self.eps = eps
        
        

        xx_pr_EV = xx_pr[:, [0], ...] * (1 + xx_pr[:, [1], ...] / xx_pr[:, [2], ...]) # Expected Value of BG distribution
        xx_pr = torch.concat((xx_pr_EV, xx_pr), dim=1).detach().clone()


        # channel 0 stats computed in log1p-space
        pr0 = torch.clamp(xx_pr[:, 0:1], min=0.0)
        pr0_sqrt = torch.sqrt(pr0)
        self.mu0 = pr0_sqrt.mean(dim=(0, 2, 3), keepdim=True)
        self.sd0 = pr0_sqrt.std(dim=(0, 2, 3), keepdim=True)

        # channels 1-3 stats in raw space
        x123 = xx_pr[:, 1:xx_pr.shape[1]]
        self.mu123 = x123.mean(dim=(0, 2, 3), keepdim=True)
        self.sd123 = x123.std(dim=(0, 2, 3), keepdim=True)

    def transform(self, x):
        x_EV = x[:, [0], ...] * (1 + x[:, [1], ...] / x[:, [2], ...])
        x = torch.concat((x_EV, x), dim=1).detach().clone()


        pr0 = torch.clamp(x[:, 0:1], min=0.0)
        pr0_sqrt = torch.sqrt(pr0)
        z0 = (pr0_sqrt - self.mu0) / (self.sd0 + self.eps)

        x123 = x[:, 1:x.shape[1]]
        z123 = (x123 - self.mu123) / (self.sd123 + self.eps)

        return torch.cat([z0, z123], dim=1).detach().clone()



    def inverse(self, z):
        z0 = z[:, 0:1]
        z123 = z[:, 1:z.shape[1]]

        pr0_sqrt = z0 * (self.sd0 + self.eps) + self.mu0
        pr0 = pr0_sqrt ** 2
        pr0 = torch.clamp(pr0, min=0.0)

        x123 = z123 * (self.sd123 + self.eps) + self.mu123

        return x123.detach().clone()
    