
save_cfg_to_final = True
stats_for_n_validation_samples = 100000 # All the validation period 
plot_n_validation_samples = 100  # set to 0 to no plot

# If continuing training:
continue_from = None
override_last_loss_pr = None
override_learningRate = None


datafolder = "data/"



import sys


MODEL_CONFIG_FILE = "CFG_Doury" # sys.argv[1]
DATA_TRANSFORMS_FILE = "CFGD_StandardTransforms4Prediction_incX1D" # sys.argv[2]

print("MODEL_CONFIG_FILE =", MODEL_CONFIG_FILE)
print("DATA_TRANSFORMS_FILE =", DATA_TRANSFORMS_FILE)


hash_cfg = True

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







xx_pr = torch.Tensor(np.load(datafolder + "x2d_train.npy"))
yy_pr = torch.Tensor(np.load(datafolder + "y_train.npy"))
xx1d_pr = torch.Tensor(np.load(datafolder + "/x1d_train.npy"))




# Transforms to y (precip):
yy_pr = yy_pr[:, None, :, :]  # add channel dimension


# Initialize transform classes
xxPrTransforms = module_data.XPrTransforms(xx_pr=xx_pr)
xx1DPrTransforms = module_data.XPrTransforms1D(xx_pr=xx1d_pr)
yy_pr_Transform = module_data.YPrTransforms(yy_pr=yy_pr)


# Apply the transforms to the data
xx_pr = xxPrTransforms.transform(xx_pr).detach().clone()
xx1d_pr = xx1DPrTransforms.transform(xx1d_pr).detach().clone()
yy_pr = yy_pr_Transform.transform(yy_pr).detach().clone()


# Ensure float32
yy_pr = yy_pr.to(dtype=torch.float32)
xx_pr = xx_pr.to(dtype=torch.float32)
xx1d_pr = xx1d_pr.to(dtype=torch.float32)


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
learningRate = module_model_config.learningRate
my_loss = module_model_config.my_loss

folder_temp = "temp/" + model_name + "/" + "pr"
folder_final = "trained_model_parameters/" + model_name
final_model_name = folder_final + "/" + "pr"


np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model_module = module_model_config.model_module





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
    early_stop_xx1d = xx1d_pr[indices]
    
    print("Indices for early-stop split: (" + str(indices.shape[0]) + "):", indices)

    mask = torch.ones(yy_pr.shape[0], dtype=torch.bool)
    mask[indices] = False
    
    indices_train = torch.arange(yy_pr.shape[0])[mask]
    
    train_yy_pr = yy_pr[mask]
    train_xx_pr = xx_pr[mask]
    train_xx1d_pr = xx1d_pr[mask]

    
    
    # Specific for EMUL-ASYM (include indices):
    # emul_asym loss is initialized with the full training dataset, computes the 
    # Indices are specifically included in [:,0,0,0] as the loss expects a very specific format: [indices, values ; height, width]
    # Although this uses a bit more memory, it's efficient and this way I can use DDPModel directly
    
    
    indices_ =  -1*torch.ones_like(early_stop_yy)
    indices_[:,0,0,0] = indices
    early_stop_yy = torch.cat((indices_, early_stop_yy), dim = 1)
    
    
    
    indices_train_ = -1*torch.ones_like(train_yy_pr)
    indices_train_[:,0,0,0] = indices_train
    train_yy_pr = torch.cat((indices_train_, train_yy_pr), dim = 1)
    
    
    
    
    # x = batch[0],  y = batch[-1], extra = batch[1:-1]
    normalized_dataset = TensorDataset(train_xx_pr, train_xx1d_pr, train_yy_pr)

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


    early_stop_dataset = TensorDataset(early_stop_xx, early_stop_xx1d, early_stop_yy)

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
    # Model Startup
    # ---------------------------------------
    model_pr = Model(
        model_module=model_module,
        ddp=use_ddp_training,
        device=device,
        world_size=world_size
    )
    
    
    if override_learningRate is not None:
        learningRate = override_learningRate

    model_pr.set_optimizer(my_optimizer, lr=learningRate)
    if my_scheduler is not None:
        model_pr.set_scheduler(my_scheduler)

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
    if continue_from is not None:
                        
        if is_main_process:
            print(f"Continuing training for PR from {continue_from}...", flush=True)
        model_pr.load_state_dict(continue_from, override_last_loss_pr)

    # ---------------------------------------
    # Train
    # ---------------------------------------
    model_pr.trainModel(
        my_loss_function = my_loss,
        train_dataloader = train_dataloader,
        earlyStop_dataloader = earlyStop_dataloader,
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
        
        
        
    if is_main_process and save_cfg_to_final:
        import shutil
        
        base = Path().absolute()

        config_path = base / f"{MODEL_CONFIG_FILE}.py"
        config_path_data = base / f"{DATA_TRANSFORMS_FILE}.py"
        
        shutil.copy2(config_path, folder_final + "/"  + config_path.name)
        shutil.copy2(config_path_data, folder_final + "/"  + config_path_data.name)
        

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
    subsample_size = int(fractionLeft4earlystop * yy_pr.shape[0])
    indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

    early_stop_yy = yy_pr[indices]
    early_stop_xx = xx_pr[indices]
    early_stop_xx1d = xx1d_pr[indices]

    stats_for_n_validation_samples = min(stats_for_n_validation_samples, subsample_size)  # Ensure we don't try to plot more samples than available in the early-stop set

    
    indices = torch.randperm(yy_pr.shape[0])[:subsample_size]

    early_stop_yy = yy_pr[indices]
    early_stop_xx = xx_pr[indices]

    random_day_indices = np.random.choice(early_stop_xx.shape[0], stats_for_n_validation_samples, replace=False)
    random_day_indices_plot = np.random.choice(random_day_indices, min(plot_n_validation_samples, stats_for_n_validation_samples), replace=False)
    
    obs_es = yy_pr_Transform.inverse(early_stop_yy[random_day_indices, ...])

    

    testDataLoader = DataLoader(
        TensorDataset(early_stop_xx[random_day_indices, ...], early_stop_xx1d[random_day_indices, ...]),
        batch_size=256,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    prediction = torch.empty((0, 1, 64, 64), dtype=torch.float32)
    for batch in testDataLoader:
        prediction = torch.cat([ prediction, model_pr.predict(batch[0], x_1D = batch[1]) ], dim=0)
    

    prediction = prediction.to(device="cpu").detach().numpy()


    import matplotlib.pyplot as plt

    nrows = len(random_day_indices_plot)

    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(4.5, 2 * nrows),
        gridspec_kw={"width_ratios": [1, 1, 0.06]},
        squeeze=False,
        constrained_layout=True
    )

    for idx, day_idx in enumerate(random_day_indices_plot):
        # idx = 0..nrows-1
        # day_idx = original index in the source dataset

        obs = obs_es[idx, 0, ...].cpu().numpy()

        pred = prediction[idx, 0, ...] 
        vmin = min(obs.min(), pred.min())
        vmax = max(obs.max(), pred.max())

        im_obs = axes[idx, 0].imshow(obs, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[idx, 0].set_title(f"Day {day_idx} - Obs")
        axes[idx, 0].set_xticks([])
        axes[idx, 0].set_yticks([])

        axes[idx, 1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[idx, 1].set_title(f"Day {day_idx} - Pred")
        axes[idx, 1].set_xticks([])
        axes[idx, 1].set_yticks([])

        cbar = fig.colorbar(im_obs, cax=axes[idx, 2])
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(folder_final + "/maps_validation.png", dpi=100, bbox_inches="tight")
    plt.close()
        
        
        
          
    # Compute Stats for the validation period:
    from functions.validateAndMeasure import validateSeriesPrecip
    stats_file_pr = os.path.join(folder_final, "stats_pr.txt")


    # General Statistics:
    obs_es = obs_es.numpy()
 
    obs_es[obs_es <= 1 ] = 0
    
    prediction_ev = prediction[:,0,...] 
    prediction_ev[prediction_ev <= 1] = 0
    
    mae_rmse = validateSeriesPrecip(prediction_ev, obs_es[:,0,...], how = "relativePRC", stats_over_wet = False)[0:2]


    with open(stats_file_pr, "w") as f:
        
                # ---- Pretty print ----
        print("\nValidation summary:\n", file=f)

        print(f"MAE  = {mae_rmse[0]:.4f}", file=f)
        print(f"RMSE = {mae_rmse[1]:.4f}\n", file=f)
        


    print(f"Statistics saved to {stats_file_pr}", flush=True)
