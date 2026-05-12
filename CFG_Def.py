


# CFG_DiffUNET.py
import torch

from diffusers import UNet2DModel
from diffusers import DDPMScheduler


model_name = "DEF"
folder_background_predictors = "final_models/main_replicateBernoulliGammaUNET_StandardTransforms4Prediction_incX1D_s0_3fb739X40b171"
seed = 0
max_epochs = 1000
timesteps = 1000
patience = 50
batch_size = 32
learningRate_pr = 4e-04
saveModelEvery = 10
dropout = 0
write_losses = True
my_optimizer = torch.optim.Adam
my_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau # can be set to None
noise_scheduler_pr = DDPMScheduler(
    num_train_timesteps = timesteps, beta_schedule = "squaredcos_cap_v2",
    clip_sample=True,
    clip_sample_range=1.0
)

use_double_precision = False
fractionLeft4earlystop = lambda dataset_size: (256*round((dataset_size*0.1)/256))/dataset_size


time_embedding_dim = 256


class wrapped_huggingUNET(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # You can add custom logic here if needed
        return super().forward(return_dict = False, *args, **kwargs)[0]



model_module_pr = wrapped_huggingUNET(
                                sample_size=torch.Size([64, 64]),
                                in_channels = 5,
                                out_channels = 1,

                                layers_per_block = 2,
                                block_out_channels = (64, 128, 256),

                                down_block_types = (
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "AttnDownBlock2D"   # attention at 32x32
                                ),
                                up_block_types = (
                                        "AttnUpBlock2D",
                                        "AttnUpBlock2D",
                                        "UpBlock2D"
                                ),

                                attention_head_dim = 8,
                                norm_num_groups = 32,
                                time_embedding_dim = time_embedding_dim
)
