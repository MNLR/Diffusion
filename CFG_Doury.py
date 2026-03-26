import torch
import torch.nn as nn
import numpy as np

from models.UNET import UNET
from models.loss_emul_asym import emul_asym
from models.aux import ConvolutionalLayer



model_name = "doury_mini_peryear"
seed = 0
max_epochs = 10000
patience = 50
batch_size = 32
fractionLeft4earlystop = lambda dataset_size: (256*round((dataset_size*0.1)/256))/dataset_size
learningRate = 0.0003
saveModelEvery = 100
dropout = 0
write_losses = True
my_optimizer = torch.optim.AdamW
weight_decay = 0.01
my_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau # can be set to None


observed = torch.Tensor(np.load("data/y_train.npy"))
groups_for_estimating_parameters = torch.arange(0, int(observed.shape[0]/365)) # all years have 365 days luckily
groups_for_estimating_parameters = groups_for_estimating_parameters.repeat_interleave(365)
my_loss = emul_asym(observed = observed, groups_for_estimating_parameters = groups_for_estimating_parameters,
                    print_distributional_info = True) 



class input_processing_block_2D(nn.Sequential):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.out_channels = int(32)
        in_channels = input_shape[1]
        kernel_size = (2,4)
        padding0 = (1,0)
        padding1 = (0,0)
        self.block = nn.Sequential(
            ConvolutionalLayer(in_channels = in_channels, out_channels = self.out_channels, 
                            kernel_size = kernel_size, padding = padding0),                                 
            ConvolutionalLayer(in_channels = self.out_channels, out_channels = self.out_channels, 
                            kernel_size = kernel_size, padding = padding1)
            )
        self.output_shape = (input_shape[0], self.out_channels,
                            input_shape[2] + 2*padding0[0] -(kernel_size[0]-1) - 1 + 1   + 2*padding1[0] - (kernel_size[0]-1) - 1 + 1,
                            input_shape[3] + 2*padding1[0] -(kernel_size[1]-1) - 1 + 1   + 2*padding1[1] - (kernel_size[1]-1) - 1 + 1
                            )

model_module = UNET(input_shape_2D = torch.Size([10, 19, 16, 22]), input_shape_1D = torch.Size([10, 43]),
                    output_channels = 1, 
                    double_resolution_times = 2, kernel_size = 2, 
                    input_processing_block_2D = input_processing_block_2D(torch.Size([10, 19, 16, 22])),
                    input_number_kernels = 32,
                    number_of_blocks = 2,
                    inputNN_neurons_list = [16, 32],
                    dropout_rate_1Dinput=dropout,
                    dropout_rate_2D=dropout)