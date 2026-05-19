# EXP_BGDiffusion_Simulate.py
#
# Simulate with the model trained in EXP_BGDiffusion.py
# - one Slurm array task = one simulation member
# - optional multi-GPU parallel simulation through DistributedSampler (per node/task)




hash_cfg = True
folder_simulations = "final_models"
max_number_of_simulations = 50  
number_of_simulations_targets = 1000
batch_size_simulation = 256




import sys

DEFAULT_MODEL_CONFIG_FILE = "CFG_TS-"
DEFAULT_DATA_TRANSFORMS_FILE = "CFGD_sqrTransforms"

MODEL_CONFIG_FILE = (
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG_FILE
)
DATA_TRANSFORMS_FILE = (
    sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_TRANSFORMS_FILE
)


import os
import importlib
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from models.DDPModel import Model


# --------------------------
# DDP setup for simulation
# --------------------------
use_ddp_simulation = int(os.environ.get("WORLD_SIZE", "1")) > 1

if use_ddp_simulation:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Keep same spirit as your previous simulation worker
    # because simulateDiffusion may communicate CPU tensors
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    device = local_rank
else:
    local_rank = 0
    rank = 0
    world_size = 1
    device = 0 if torch.cuda.is_available() else "cpu"

is_main_process = (rank == 0)

print(f"[simulate] rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)


# --------------------------
# Slurm array index = simulation member
# --------------------------
slurm_step_i = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if is_main_process:
    print(f"SLURM_ARRAY_TASK_ID = {slurm_step_i}", flush=True)


# --------------------------
# Import config modules
# --------------------------
module_model_config = importlib.import_module(MODEL_CONFIG_FILE)
module_data = importlib.import_module(DATA_TRANSFORMS_FILE)


# --------------------------
# Hash logic copied from training
# --------------------------
def file_hash(path, n=6):
    h = hashlib.sha256()
    with open(path, "rb") as f:
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
xx_pr = torch.Tensor(np.load(module_model_config.folder_background_predictors  + "/prediction_train.npy"))
yy_pr = torch.Tensor(np.load("data/" + "/y_train.npy"))
yy_pr = yy_pr[:, None, :, :]  # ensure yy_pr has shape (N, 1, H, W) for the transforms (and later for the model)



# Initialize transform classes
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr)

del(xx_pr)
del(yy_pr)


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
noise_scheduler_pr = module_model_config.noise_scheduler_pr
model_module = module_model_config.model_module_pr

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



# --------------------------
# Load / prepare test set
# --------------------------
# Replace this loader if needed.
xx_test_pr_raw = torch.Tensor(np.load(module_model_config.folder_background_predictors  + "/prediction_test.npy"))



xx_test_pr = xxPrTransforms.transform(xx_test_pr_raw)
xx_test_pr = xx_test_pr.to(dtype=torch.float32)


# --------------------------
# Simulate
# --------------------------
folder_simulations = os.path.join(folder_simulations, "simulations")
os.makedirs(folder_simulations, exist_ok=True)





# For specific days:
dateS_ids = np.load("data" + "/target_dates.npy")


noise_scheduler_pr.set_timesteps(module_model_config.timesteps, device=model_pr.device)

random_noise = torch.randn(
    (dateS_ids.shape[0] * number_of_simulations_targets, 1, xx_test_pr.shape[2], xx_test_pr.shape[3])
)
conditionings = xx_test_pr[dateS_ids, ...]
conditionings = conditionings.repeat_interleave(number_of_simulations_targets, dim=0)
index = torch.arange(conditionings.shape[0], dtype=torch.long)


testDataLoader_days = DataLoader(
    TensorDataset(random_noise, conditionings, index),
    batch_size=batch_size_simulation,
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

if is_main_process:
    print(f"Simulating for {dateS_ids.shape[0]} days with {number_of_simulations_targets} simulations each...", flush=True)

simulations_targets = model_pr.simulateDiffusion(
    testDataLoader_days,
    noise_scheduler_pr,
    transformSimulation=yy_pr_Transform.inverse
)

if is_main_process:

    simulations_targets = torch.tensor(
        simulations_targets.reshape(
            dateS_ids.shape[0],
            number_of_simulations_targets,
            xx_test_pr.shape[2],
            xx_test_pr.shape[3]
        )
    )


np.save(folder_simulations + "/simulations_days.npy", simulations_targets)


del(random_noise)
del(conditionings)
del(testDataLoader_days)
del(simulations_targets)





# The complete set of simulations:

n_test = xx_test_pr.shape[0]
h = xx_test_pr.shape[2]
w = xx_test_pr.shape[3]

if is_main_process:
    print(f"Loaded test set with shape {tuple(xx_test_pr.shape)}", flush=True)


# --------------------------
# Build distributed test dataloader
# --------------------------
sample = torch.randn(n_test, 1, h, w)
index = torch.arange(n_test, dtype=torch.long)

normalized_dataset_test = TensorDataset(sample, xx_test_pr, index)

if use_ddp_simulation:
    test_sampler = DistributedSampler(
        normalized_dataset_test,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
else:
    test_sampler = None

test_dataloader = DataLoader(
    normalized_dataset_test,
    batch_size=int(np.ceil(n_test / world_size)),
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    sampler=test_sampler
)


keep_simulating = (len(os.listdir(folder_simulations)) < max_number_of_simulations)

while keep_simulating:
    if is_main_process:
        print(f"Current number of simulations: {len(os.listdir(folder_simulations))}.", flush=True)

    noise_scheduler_pr.set_timesteps(module_model_config.timesteps, device=model_pr.device)

    # Generate random string based on current time in nanosecond
    # I use this for the name!
    seed_simulation = time.time_ns() % 10_000_000
    np.random.seed(seed_simulation) 
    torch.manual_seed(seed_simulation)  



    simname = os.path.join(
        folder_simulations,
        "simulation_" + str(seed_simulation) + str(slurm_step_i) + ".npy"
    )

    if is_main_process:
        print("\nProducing simulation " + simname + "...\n", flush=True)



    model_pr.simulateDiffusion(
        test_dataloader,
        noise_scheduler_pr,
        sub_batch_size = batch_size_simulation,
        transformSimulation=yy_pr_Transform.inverse,
        file_name=simname
    )

    if is_main_process:
        print(f"Simulation saved to {simname}", flush=True)

    keep_simulating = (len(os.listdir(folder_simulations)) < max_number_of_simulations)



if use_ddp_simulation:
    dist.barrier()
    dist.destroy_process_group()



