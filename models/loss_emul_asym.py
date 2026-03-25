import torch
import torch.nn as nn
from scipy.stats import gamma

    
        
class emul_asym(nn.Module):
# predicted. Structure:(N, :) 
# observed. Structure:(N, :)

    def __init__(self, observed, device = None,  wet_threshold = 1, print_distributional_info = False):
        super().__init__()
        self.device = device

        self.cdf_target = torch.zeros(observed.shape)
        self.cdf_target_batch = None
        for i in range(observed.shape[1]):
            for j in range(observed.shape[2]):
                data = observed[:, i, j]
                data = data[data > wet_threshold]  # Filter values over wet_threshold mm
                if len(data) > 0:
                    shape, loc, scale = gamma.fit(data, floc = wet_threshold)  # Fit gamma distribution
                    
                else:
                    raise ValueError(f"No data points above wet_threshold at grid point ({i}, {j})")
                
                if (print_distributional_info):
                    print("Grid point (" + str(i) + "," + str(j) + "): shape = " + str(shape) +
                          " scale = " + str(scale) + " loc = " + str(loc))
                    
                # Now compute CDF(target) for each gridpoint:
                self.cdf_target[:, i, j] = torch.pow( torch.Tensor(gamma.cdf(x = observed[:, i, j], 
                                                                             a = shape, 
                                                                             scale = scale,
                                                                             loc = wet_threshold)),
                                                     2)
        
        self.cdf_target = self.cdf_target.unsqueeze(1)  # Make it (N, 1, height, width)
        
        if self.device is not None:
            self.cdf_target = self.cdf_target.to(self.device)
                                            
        
    def forward(self, observed, predicted):
        

        # Warning expects a very specific format: [indices, values ; height, width]
        indices_of_batch = observed[:,0,...][:, 0,0].clone().detach().long()
        observed_ = observed[:,1,...].clone().detach()
        
        if predicted.shape != observed_.shape:
            observed_ = observed_.unsqueeze(1)
            
            
        if self.device is None:
            self.cdf_target_batch = (self.cdf_target[(indices_of_batch.to("cpu")), ...]).clone().detach()
            self.cdf_target_batch = self.cdf_target_batch.to(observed_.device)


        # This won't work unless send_cdftarget_to_device has been called before:
        loss = torch.abs( observed_ - predicted ) + \
            self.cdf_target_batch*torch.clamp(observed_ - predicted , min = 0)
        loss = loss.mean()

        return(loss)