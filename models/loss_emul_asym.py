import torch
import torch.nn as nn
from scipy.stats import gamma

    
    
  
class emul_asym(nn.Module):
# predicted. Structure:(N, :) 
# observed. Structure:(N, :)
# The forward method requires indices and observed values.
# The indices are necesary to get the corresponding CFD

    def __init__(self, observed,
                 groups_for_estimating_parameters = None,
                 wet_threshold = 1, print_distributional_info = False):
        super().__init__()

        if groups_for_estimating_parameters is not None:
            assert groups_for_estimating_parameters.dtype == torch.long, "groups_for_estimating_parameters must be of type torch.long"
            number_of_groups = groups_for_estimating_parameters.max() + 1
        
        self.cdf_target = torch.zeros(observed.shape)
        self.cdf_target_batch = None
        for i in range(observed.shape[1]):
            for j in range(observed.shape[2]):
                data = observed[:, i, j]
                
                if groups_for_estimating_parameters is not None:
                    shape, loc, scale = 0, 0, 0
                    for g in range(number_of_groups):
                        data_group = data[groups_for_estimating_parameters == g]
                        
                        data_group = data_group[data_group > wet_threshold]  # Filter values over wet_threshold mm
                        
                        if len(data_group) > 0:
                            shape_g, loc_g, scale_g = gamma.fit(data_group, floc = wet_threshold)  # Fit gamma distribution
                            shape += shape_g
                            loc += loc_g
                            scale += scale_g
                        else:
                            raise ValueError(f"No data points above wet_threshold at grid point ({i}, {j}) for group {g}")
                        

                        # Now compute CDF(target) for each gridpoint:
                        
                    shape /= number_of_groups
                    loc /= number_of_groups
                    scale /= number_of_groups
                    
                else:
                    data = data[data > wet_threshold]  # Filter values over wet_threshold mm
                    if len(data) > 0:
                        shape, loc, scale = gamma.fit(data, floc = wet_threshold)  # Fit gamma distribution
                    else:
                        raise ValueError(f"No data points above wet_threshold at grid point ({i}, {j})")
                    
                        
                if (print_distributional_info):
                    print("Grid point (" + str(i) + "," + str(j) + "): shape = " + str(shape) +
                        " scale = " + str(scale) + " loc = " + str(loc))
                                        
                    
                self.cdf_target[:, i, j] = torch.pow( torch.Tensor(gamma.cdf(x = observed[:, i, j], 
                                                                            a = shape, 
                                                                            scale = scale,
                                                                            loc = wet_threshold)),
                                                    2)
    
        self.cdf_target = self.cdf_target.unsqueeze(1)  # Make it (N, 1, height, width)
        
                      
        
    def forward(self, observed, predicted):
        

        # Warning expects a very specific format: [indices, values ; height, width]
        indices_of_batch = observed[:,0,...][:, 0,0].clone().detach().long()
        observed_ = observed[:,1,...].clone().detach()
        
        if predicted.shape != observed_.shape:
            observed_ = observed_.unsqueeze(1)
            
            
        self.cdf_target_batch = (self.cdf_target[(indices_of_batch.to("cpu")), ...]).clone().detach()
        self.cdf_target_batch = self.cdf_target_batch.to(observed_.device)


        # This won't work unless send_cdftarget_to_device has been called before:
        loss = torch.abs( observed_ - predicted ) + \
            self.cdf_target_batch*torch.clamp(observed_ - predicted , min = 0)
        loss = loss.mean()

        return(loss)