MODEL_CONFIG_FILE = "CFG_BernoulliGammaUNET"  # Common to all experiments and domains.
DATA_TRANSFORMS_FILE = "CFGD_StandardTransforms4Prediction_incX1D" 
hash_cfg = True
folder_simulations = "final_models"
max_number_of_simulations = 50  
datafolder = "data/"
batch_size_simulation = 256
save_some_stats = True
save_prediction_as_diffusion_background = True

import os
import importlib
from functions.simulateBernoulliGamma import simulate_bernoulli_gamma
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from models.DDPModel import Model


local_rank = 0
rank = 0
world_size = 1
device = 0

is_main_process = (rank == 0)

print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)



module_model_config = importlib.import_module(MODEL_CONFIG_FILE)
module_data = importlib.import_module(DATA_TRANSFORMS_FILE)



if hash_cfg:
    from pathlib import Path
    import hashlib

    def file_hash(path, n=6):
        h = hashlib.sha256()
        with open(path, "rb") as f:  # binary mode (important)
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:n]

    if hash_cfg:
        base = Path().absolute()

        config_path = base / f"{MODEL_CONFIG_FILE}.py"
        config_path_data = base / f"{DATA_TRANSFORMS_FILE}.py"

        hash_model = file_hash(config_path)
        hash_data = file_hash(config_path_data)

        unique_identifier_hash = f"{hash_model}X{hash_data}"


# --------------------------
# Load training data to fit transforms exactly as in training
# --------------------------

xx_pr = torch.Tensor(np.load(datafolder + "x2d_train.npy"))
yy_pr = torch.Tensor(np.load(datafolder + "y_train.npy"))
xx1d_pr = torch.Tensor(np.load(datafolder + "x1d_train.npy"))


pr_shape = yy_pr.shape


# Transforms to y (precip):
yy_pr = yy_pr[:, None, :, :]  # add channel dimension

# Initialize transform classes
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr)
xx1DPrTransforms = module_data.XPrTransforms1D(xx_pr=xx1d_pr)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr)









# --------------------------
# Rebuild trained model name exactly as in training
# --------------------------
model_name = (
    module_model_config.model_name
    + MODEL_CONFIG_FILE.split("_", 1)[1]
    + "_"
    + DATA_TRANSFORMS_FILE.split("_", 1)[1]
    + "_s"
    + str(module_model_config.seed)
)

if hash_cfg:
    model_name += "_" + unique_identifier_hash

folder_final = "trained_model_parameters/" + model_name
folder_simulations = folder_simulations + "/" + model_name
final_model_name = folder_final + "/pr"

if is_main_process:
    print(f"Loading model from {final_model_name}", flush=True)


# --------------------------
# Load model
# --------------------------
model_module = module_model_config.model_module

# Keep ddp=False here.
# Parallelism for simulation is handled by DistributedSampler + one process per GPU.
model_pr = Model(
    model_module=model_module,
    ddp=False,
    device=device,
    world_size=world_size
)
model_pr.load_state_dict(final_model_name)

if is_main_process:
    print("Model loaded.", flush=True)





if save_prediction_as_diffusion_background:
    xx_pr = xxPrTransforms.transform(xx_pr)
    xx1d_pr = xx1DPrTransforms.transform(xx1d_pr)

    TensorDataset_train = TensorDataset(xx_pr, xx1d_pr)
    train_dataloader = DataLoader(TensorDataset_train, batch_size = batch_size_simulation,
                                shuffle=False, drop_last=False)

    prediction_train = torch.empty((0, 3, pr_shape[-2], pr_shape[-1]), dtype=torch.float32)
    for batch in train_dataloader:
        prediction_train = torch.cat([prediction_train, model_pr.predict(batch[0], x_1D = batch[1])], dim=0)
                
    np.save(folder_simulations + "/prediction_train.npy", prediction_train.cpu().numpy())



del(xx_pr)
del(xx1d_pr)
del(TensorDataset_train)
del(train_dataloader)
del(prediction_train)



xx2d_test = torch.Tensor(np.load(datafolder + "x2d_test.npy"))
xx1d_test = torch.Tensor(np.load(datafolder + "x1d_test.npy"))



xx2d_test = xxPrTransforms.transform(xx2d_test)
xx1d_test = xx1DPrTransforms.transform(xx1d_test)



TensorDataset_test = TensorDataset(xx2d_test, xx1d_test)
test_dataloader = DataLoader(TensorDataset_test, batch_size = batch_size_simulation,
                             shuffle=False, drop_last=False)

prediction = torch.empty((0, 3, pr_shape[-2], pr_shape[-1]), dtype=torch.float32)
for batch in test_dataloader:
    prediction = torch.cat([prediction, model_pr.predict(batch[0], x_1D = batch[1])], dim=0)
            



if save_prediction_as_diffusion_background:
    np.save(folder_simulations + "/prediction_test.npy", prediction.cpu().numpy())



# --------------------------
# Simulate
# --------------------------
folder_simulations = os.path.join(folder_simulations, "simulations")
os.makedirs(folder_simulations, exist_ok=True)





for sim_name in range(50):
    simulation = simulate_bernoulli_gamma(p = prediction[:, 0, ...],
                                          shape = prediction[:,1, ...], rate = prediction[:,2, ...],
                                          localize = 1).unsqueeze(1)

    simulation = simulation.numpy()
    np.save(folder_simulations + "/simulation_" + str(sim_name) + ".npy", simulation)




dateS_ids = np.load(datafolder + "/target_dates.npy")
simulations_targets = np.zeros((9, 1000, 64, 64))

for i in range(1000):
    simulations_targets[:,i,...] = simulate_bernoulli_gamma(p = prediction[dateS_ids, 0, ...],
                                                            shape = prediction[dateS_ids,1, ...],
                                                            rate = prediction[dateS_ids,2, ...],
                                          localize = 1)


np.save(folder_simulations + "/simulations_days.npy", simulations_targets)




