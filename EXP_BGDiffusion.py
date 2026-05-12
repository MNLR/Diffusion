# EXP_BGDiffusion.py
#
# Train the model
# Optional multi-GPU parallel simulation through DistributedSampler (per node/task)


hash_cfg = True
save_cfg_to_final = True
stats_for_n_validation_samples = 1024
number_of_simulations_to_try = 32
plot_n_validation_samples = 15  # set to 0 to no plot

# If continuing training:
continue_from_pr = None
override_last_loss_pr = None
override_learningRate_pr = None






import sys

DEFAULT_MODEL_CONFIG_FILE = "CFG_Def"
DEFAULT_DATA_TRANSFORMS_FILE = "CFGD_sqrTransforms"

MODEL_CONFIG_FILE = (
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_CONFIG_FILE
)
DATA_TRANSFORMS_FILE = (
    sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATA_TRANSFORMS_FILE
)


import os
import importlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from models.DDPModel import Model



# --------------------------
# DDP setup
# --------------------------
use_ddp_training = int(os.environ.get("WORLD_SIZE", "1")) > 1

if use_ddp_training:
    # GPU DDP training -> NCCL
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = local_rank
else:
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
# Load data
# --------------------------
xx_pr = torch.Tensor(np.load(module_model_config.folder_background_predictors  + "/prediction_train.npy"))
yy_pr = torch.Tensor(np.load("data/" + "/y_train.npy"))
yy_pr = yy_pr[:, None, :, :]  # ensure yy_pr has shape (N, 1, H, W) for the transforms (and later for the model)


# Initialize transform classes
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr)


# Apply the transforms to the data
xx_pr = xxPrTransforms.transform(xx_pr)
yy_pr = yy_pr_Transform.transform(yy_pr)




# Ensure float32
yy_pr = yy_pr.to(dtype=torch.float32)
xx_pr = xx_pr.to(dtype=torch.float32)



# --------------------------
# Load config parameters
# --------------------------
model_name = module_model_config.model_name + MODEL_CONFIG_FILE.split("_", 1)[1] + "_" + DATA_TRANSFORMS_FILE.split("_", 1)[1] + "_s" +  str(module_model_config.seed)
if hash_cfg:
    model_name += "_" + unique_identifier_hash



fractionLeft4earlystop = module_model_config.fractionLeft4earlystop(xx_pr.shape[0])

seed = module_model_config.seed
max_epochs = module_model_config.max_epochs
batch_size = module_model_config.batch_size
patience = module_model_config.patience
saveModelEvery = module_model_config.saveModelEvery
write_losses = module_model_config.write_losses
my_optimizer = module_model_config.my_optimizer
my_scheduler = module_model_config.my_scheduler

learningRate_pr = module_model_config.learningRate_pr

folder_temp = "temp/" + model_name + "/" + "pr"
folder_final = "trained_model_parameters/" + model_name
final_model_name = folder_final + "/" + "pr"



# ---------------------------------------
# Folders only on rank 0
# ---------------------------------------
if is_main_process:
    if not os.path.exists(folder_temp):
        os.makedirs(folder_temp)
    else:
        print(f"Warning: Folder {folder_temp} already exists. Files may be overwritten.")

    if not os.path.exists(folder_final):
        os.makedirs(folder_final)
    else:
        print(f"Warning: Folder {folder_final} already exists. Files may be overwritten.")

if is_main_process and save_cfg_to_final:
    import shutil
    
    base = Path().absolute()

    config_path = base / f"{MODEL_CONFIG_FILE}.py"
    config_path_data = base / f"{DATA_TRANSFORMS_FILE}.py"
    
    shutil.copy2(config_path, folder_final + "/"  + config_path.name)
    shutil.copy2(config_path_data, folder_final + "/"  + config_path_data.name)

if use_ddp_training:
    dist.barrier()





np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

noise_scheduler_pr = module_model_config.noise_scheduler_pr
model_module = module_model_config.model_module_pr





if use_ddp_training:
    exists_list = [os.path.exists(final_model_name) if is_main_process else None]
    dist.broadcast_object_list(exists_list, src=0)
    model_already_exists = exists_list[0]
else:
    model_already_exists = os.path.exists(final_model_name)





if not model_already_exists:

    if is_main_process:
        print("Starting training...", flush=True)

    # ---------------------------------------
    # Split train / early-stop IDENTICALLY on all ranks
    # ---------------------------------------
    torch.manual_seed(seed)
    subsample_size = int(fractionLeft4earlystop * yy_pr.shape[0])
    indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

    early_stop_yy = yy_pr[indices]
    early_stop_xx = xx_pr[indices]
    
    print("Indices for early-stop split: (" + str(indices.shape[0]) + "):", indices)

    mask = torch.ones(yy_pr.shape[0], dtype=torch.bool)
    mask[indices] = False
    train_yy_pr = yy_pr[mask]
    train_xx_pr = xx_pr[mask]

    normalized_dataset = TensorDataset(train_yy_pr, train_xx_pr)

    if use_ddp_training:
        train_sampler = DistributedSampler(
            normalized_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed
        )
    else:
        train_sampler = None


    # Keep early-stop simple. If your DDPModel explicitly supports distributed
    # validation reduction, you can also replace this by a DistributedSampler.
    early_stop_dataset = TensorDataset(early_stop_yy, early_stop_xx)

    train_dataloader = DataLoader(
        normalized_dataset,
        batch_size = int(batch_size / world_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        pin_memory=True
    )

    earlyStop_dataloader = DataLoader(
        early_stop_dataset,
        batch_size=256,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    # ---------------------------------------
    # Model
    # ---------------------------------------
    model_pr = Model(
        model_module=model_module,
        ddp=use_ddp_training,
        device=device,
        world_size=world_size
    )
    
    
    if override_learningRate_pr is not None:
        learningRate_pr = override_learningRate_pr

    model_pr.set_optimizer(my_optimizer, lr=learningRate_pr)
    if my_scheduler is not None:
        model_pr.set_scheduler(my_scheduler)


    if use_ddp_training:
        dist.barrier()
    # ---------------------------------------
    # Parameter count only on rank 0
    # ---------------------------------------
    if is_main_process:
        model_for_count = model_pr.model.module if hasattr(model_pr.model, "module") else model_pr.model
        total_params = sum(p.numel() for p in model_for_count.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters in the model: {total_params}", flush=True)

    # ---------------------------------------
    # Optional checkpoint resume
    # ---------------------------------------
    if continue_from_pr is not None:   
        if is_main_process:
            print(f"Continuing training for PR from {continue_from_pr}...", flush=True)
        model_pr.load_state_dict(continue_from_pr, override_last_loss_pr)

    # ---------------------------------------
    # Train
    # ---------------------------------------
    model_pr.trainDiffusionModel(
        train_dataloader=train_dataloader,
        noise_scheduler=noise_scheduler_pr,
        earlyStop_dataloader=earlyStop_dataloader,
        max_epochs=max_epochs,
        patience=patience,
        saveModelEvery=saveModelEvery,
        write_losses=write_losses,
        folder_temp=folder_temp,
        final_model_name=final_model_name,
        verbose=is_main_process
    )



    if use_ddp_training:
        dist.barrier()
        
        

else:
    if is_main_process:
        print(f"Model {final_model_name}.pt already exists. Skipping training.", flush=True)




# --------------------------
# Tear down DDP before diagnostics
if use_ddp_training:
    dist.barrier()
    dist.destroy_process_group()





# --------------------------
# Post-training diagnostics only on rank 0
# --------------------------
if is_main_process:

    # Load a plain single-GPU / non-DDP model for simulation and plotting
    model_pr = Model(
        model_module=model_module,
        ddp=False,
        device=0,
        world_size=1
    )
    model_pr.load_state_dict(final_model_name)


    # Recreate the same early-stop split used above
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    subsample_size = int(fractionLeft4earlystop * yy_pr.shape[0])
    stats_for_n_validation_samples = min(stats_for_n_validation_samples, subsample_size)  # Ensure we don't try to plot more samples than available in the early-stop set
    
    indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

    early_stop_yy = yy_pr[indices]
    early_stop_xx = xx_pr[indices]

    random_day_indices = np.random.choice(early_stop_xx.shape[0], stats_for_n_validation_samples, replace=False)

    obs_es = yy_pr_Transform.inverse(early_stop_yy[random_day_indices, ...])

    noise_scheduler_pr.set_timesteps(module_model_config.timesteps, device=model_pr.device)

    random_noise = torch.randn(
        (stats_for_n_validation_samples * number_of_simulations_to_try, 1, early_stop_xx.shape[2], early_stop_xx.shape[3])
    )
    conditionings = early_stop_xx[random_day_indices, ...]
    conditionings = conditionings.repeat_interleave(number_of_simulations_to_try, dim=0)
    index = torch.arange(conditionings.shape[0], dtype=torch.long)

    testDataLoader = DataLoader(
        TensorDataset(random_noise, conditionings, index),
        batch_size=256,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    simus_ = model_pr.simulateDiffusion(
        testDataLoader,
        noise_scheduler_pr,
        transformSimulation=yy_pr_Transform.inverse
    )

    simulations_days = torch.tensor(
        simus_.reshape(
            stats_for_n_validation_samples,
            number_of_simulations_to_try,
            early_stop_xx.shape[2],
            early_stop_xx.shape[3]
        )
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        stats_for_n_validation_samples,
        number_of_simulations_to_try + 2,
        figsize=(2 * (number_of_simulations_to_try + 1) + 0.5, 2 * stats_for_n_validation_samples),
        gridspec_kw={
            "width_ratios": [1] * (number_of_simulations_to_try + 1) + [0.06]
        },
        squeeze=False,
        constrained_layout=True
    )

    for idx, day_idx in enumerate(random_day_indices):
        obs = obs_es[idx, 0, ...].numpy()

        vmin = obs.min()
        vmax = obs.max()
        for sim_idx in range(number_of_simulations_to_try):
            pred = simulations_days[idx, sim_idx, ...].numpy()
            vmin = min(vmin, pred.min())
            vmax = max(vmax, pred.max())

        im_obs = axes[idx, 0].imshow(obs, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[idx, 0].set_title(f"Day {day_idx} - Obs")
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])

        for sim_idx in range(number_of_simulations_to_try):
            pred = simulations_days[idx, sim_idx, ...].numpy()
            axes[idx, sim_idx + 1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[idx, sim_idx + 1].set_title(f"Day {day_idx} - Sim {sim_idx + 1}")
            axes[idx, sim_idx + 1].set_xticks([])
            axes[idx, sim_idx + 1].set_yticks([])

        cbar = fig.colorbar(im_obs, cax=axes[idx, -1])
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(folder_final + "/maps_validation.png", dpi=100, bbox_inches="tight")
    plt.close()

    
    
    # Compute Stats for the validation period:
    from functions.validateAndMeasure import validateSeriesPrecip, measureDistributionalPrecip
    stats_file_pr = os.path.join(folder_final, "stats_pr.txt")





    # General Statistics:
    simulations_days = simulations_days.numpy()
    obs_es = obs_es.numpy()
    
    validation_ = np.zeros((number_of_simulations_to_try, 6))  # mean, mean_wet, sd, p95, p99 for each day
    validation_obs = np.zeros(6)  # mean, mean_wet, sd, p95, p99 for obs (same for all simulations)
    
    validation_obs[0] = measureDistributionalPrecip(obs_es, stats_over_wet = False)[0]*100
    mswo = measureDistributionalPrecip(obs_es, stats_over_wet = True)
    validation_obs[1] = mswo[1]  # mean_wet
    validation_obs[2] = mswo[2]  # sd_wet
    validation_obs[3] = mswo[7]  # p50_wet
    validation_obs[4] = mswo[9]  # p95_wet
    validation_obs[5] = mswo[10] # p99_wet
    
    for i in range(number_of_simulations_to_try):
        # (WET_DAYS 0, MEAN 1, SD 2, VAR 3, P01 4, P05 5, P25 6, P50 7, P75 8, P95 9, P99 10)
        msd = measureDistributionalPrecip(simulations_days[:,i, ...], stats_over_wet = False)[0]*100
        msw = measureDistributionalPrecip(simulations_days[:, i, ...], stats_over_wet = True)
        validation_[i] = (msd, msw[1], msw[2], msw[7], msw[9], msw[10])  # mean_wet, sd_wet, p50_wet, p95_wet, p99_wet
    
        
    validation_.max(axis=0)
        
    # ---- Aggregate statistics over simulations ----
    val_mean = validation_.mean(axis=0)
    val_std  = validation_.std(axis=0)
    val_min  = validation_.min(axis=0)
    val_max  = validation_.max(axis=0)

    # ---- Stack into table ----
    table = np.vstack([
        validation_obs,
        val_mean,
        val_std,
        val_min,
        val_max
    ])


    # ---- Optional: labels ----
    row_labels = ["obs", "mean_sim", "std_sim", "min_sim", "max_sim"]
    col_labels = ["wet day %", "mean_wet", "sd_wet", "p50_wet", "p95_wet", "p99_wet"]

    simulations_days[simulations_days <= 1 ] = 0
    obs_es[obs_es <= 1 ] = 0
    
    mae_rmse = validateSeriesPrecip(simulations_days.mean(1), obs_es[:,0,...], how = "relativePRC", stats_over_wet = False)[0:2]


    with open(stats_file_pr, "w") as f:
        
                # ---- Pretty print ----
        print("\nValidation summary:\n", file=f)

        print(f"MAE  = {mae_rmse[0]:.4f}", file=f)
        print(f"RMSE = {mae_rmse[1]:.4f}\n", file=f)

        # header
        print(f"{'':12s}" + "".join([f"{c:>12s}" for c in col_labels]), file=f)

        # rows
        for i, row in enumerate(table):
            print(f"{row_labels[i]:12s}" + "".join([f"{v:12.3f}" for v in row]), file=f)        
        
        
            
        
        
        
        print("\n \n \nStatistics for the observed days:", file=f)
            

        for idx, day_idx in enumerate(random_day_indices):
            obs = obs_es[idx, 0, ...]

            print(
                f"Day {day_idx} - Obs: "
                f"mean={obs.mean():.3f}, "
                f"min={obs.min():.3f}, "
                f"max={obs.max():.3f}",
                file=f
            )

            sim_day = simulations_days[idx]
            sim_means = sim_day.mean((1, 2))
            sim_min = sim_day.min(axis=1).min(1).mean()
            sim_max = sim_day.max(axis=1).max(1).mean()

            print(
                f"Day {day_idx} - Sim: "
                f"mean={sim_means.mean():.3f} "
                f"({sim_means.min():.3f} - {sim_means.max():.3f}), "
                f"min={sim_min:.3f}, "
                f"max={sim_max:.3f}",
                file=f
            )

            print("---", file=f)

    print(f"Statistics saved to {stats_file_pr}", flush=True)
