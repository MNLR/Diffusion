import torch.nn as nn
import torch
from copy import deepcopy
import datetime
import time
from tqdm import tqdm
import numpy as np
import inspect

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import ReduceOp, gather


class Model:
    

            

    def __init__(self, model_module: nn.Module,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ddp: bool = False, # If True, wrap the model in DDP.
                 world_size: int = 1                 
                 ):
        
        self.optimizer = None
        self.scheduler = None

        distributed_is_initialized = (
            dist.is_available() and dist.is_initialized()
        )
        self.rank = dist.get_rank() if distributed_is_initialized else 0
        self.world_size = (
            dist.get_world_size() if distributed_is_initialized else world_size
        )
        self.is_main_process = self.rank == 0
        self.ddp = bool(ddp and self.world_size > 1)

        if self.ddp and not distributed_is_initialized:
            raise RuntimeError(
                "DDP requires an initialized torch.distributed process group."
            )

        if isinstance(device, int):
            if device < 0 or not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda", device)
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"CUDA device {self.device} was requested, but CUDA is unavailable."
                    )
                if self.device.index is None:
                    self.device = torch.device(
                        "cuda", torch.cuda.current_device()
                    )

        self.bestLoss = torch.inf
        self.losses = None
        self.lossesEpoch = None
        self.lossesTest = None
        self.trained = False


        # get all additional arguments of the forward method, except self and x:
        self.additional_forward_args = inspect.getfullargspec(model_module.forward)[0][2:] 


        self.model = model_module.to(self.device)
        

        
        if self.ddp:
            if self.device.type == "cuda":
                self.model = DDP(
                    self.model,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )
            else:
                self.model = DDP(self.model)
            self.bestmodelStateDict = deepcopy( self.model.module.state_dict() )
        else:
            self.bestmodelStateDict = deepcopy( self.model.state_dict() )
            
            
            
    @staticmethod
    def _validate_training_args(
        max_epochs,
        saveModelEvery,
        write_losses,
        folder_temp,
    ):
        if (
            isinstance(max_epochs, bool)
            or not isinstance(max_epochs, (int, np.integer))
            or max_epochs <= 0
        ):
            raise ValueError(
                "max_epochs must be a positive finite integer."
            )

        if (
            (saveModelEvery != torch.inf or write_losses)
            and folder_temp is None
        ):
            raise ValueError(
                "folder_temp must be set when saving "
                "checkpoints or losses."
            )
            

    def _model_module(self):
        return self.model.module if self.ddp else self.model

    def _load_model_state_dict(self, state_dict):
        """Restore in-memory weights without changing checkpoint metadata."""
        self._model_module().load_state_dict(state_dict)

    def _epoch_loss(self, loss_sum, sample_count):
        """Global sample-weighted mean of scalar batch-mean losses.

        Loss functions must return a mean over samples (and, for fixed-size
        fields, their elements). Counts refer to samples actually processed,
        including any padding introduced by a distributed sampler.
        """
        totals = torch.tensor(
            [loss_sum, sample_count], device=self.device, dtype=torch.float64
        )
        if self.ddp:
            dist.all_reduce(totals, op=ReduceOp.SUM)
        if totals[1].item() == 0:
            raise ValueError("train_dataloader is empty.")
        return (totals[0] / totals[1]).item()


        
    def set_optimizer(self, optimizer, **kwargs):
        if self.optimizer is not None:
            print("Optimizer changed. Rerun set_scheduler(). It has been set to None.")
            self.scheduler = None
            
        self.optimizer = optimizer(params=self.model.parameters(), **kwargs)
        
        
        
    def set_scheduler(self, scheduler, **kwargs):
        if self.optimizer is not None:
            self.scheduler = scheduler(optimizer = self.optimizer, **kwargs)
        else:
            raise ValueError("Optimizer must be set first")
        
        
        
    def train_iter(self, x, y, loss_function, **forward_kwargs) -> torch.Tensor:

        training_mode = self.model.training
        if not training_mode:
            self.model = self.model.train()
        
        x = x.to(self.device)
        
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        y = y.to(self.device)
        
        
        # Perform one step of training:        
        self.optimizer.zero_grad()
                    
        output = self.model(x, **forward_kwargs)
        
        loss = loss_function(y, output)
            
        loss.backward()
        
        self.optimizer.step()
        
        
        self.model = self.model.train(mode = training_mode)
        
        return loss.item()
    
    
    
    def _test_iter_with_model(
        self,
        model,
        x,
        y,
        loss_function,
        verbose=True,
        **forward_kwargs,
    ) -> torch.Tensor:
        training_mode = model.training
        if training_mode:
            model.eval()
        
        x = x.to(self.device)
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        
        y = y.to(self.device)

        
        with torch.no_grad():
            output = model(x, **forward_kwargs)
            loss = loss_function(y, output)
                                
        if verbose:
            print(f"Loss Test (acc): {loss}")                                
            
            
        model.train(mode=training_mode)
        
    
        return loss.item()


    def test_iter(self, x, y, loss_function,
                  verbose = True, **forward_kwargs) -> torch.Tensor:
        return self._test_iter_with_model(
            self.model,
            x,
            y,
            loss_function,
            verbose=verbose,
            **forward_kwargs,
        )
    
    
    
    def predict(self, x, **forward_kwargs):
        
        original_device = x.device
        
        training_mode = self.model.training
        if training_mode:
           self.model = self.model.eval() 
        
        x = x.to(self.device)
                
        for key in forward_kwargs:
            forward_kwargs[key] = forward_kwargs[key].to(self.device)
        
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, **forward_kwargs)
            
            
        self.model = self.model.train(mode = training_mode)   
    
        
        return output.to(original_device)
    
    
    
    def updateBestModelandLoss(self, loss):
        
        if loss < self.bestLoss:
            self.bestmodelStateDict = deepcopy(self._model_module().state_dict())
            
            self.bestLoss = loss
        
    

    
    def scheduler_step(self, loss, verbose = True):
        if (self.scheduler is not None):
            last_lr = self.scheduler.get_last_lr()
            self.scheduler.step(loss)
            if verbose and self.is_main_process:
                if last_lr != self.scheduler.get_last_lr():
                    print(f"Scheduler changed LR from {last_lr} to {self.scheduler.get_last_lr()}")
        
        else:
            if verbose and self.is_main_process:
                print("Scheduler not set. Ignoring scheduler step.")
                
                
                

    def load_state_dict(self, state_dict, last_loss = None, weights_only = True):
        
        loaded_state_dict = torch.load(
            state_dict,
            weights_only=weights_only,
            map_location=self.device,
        )
        
        self._load_model_state_dict(loaded_state_dict)
        self.bestmodelStateDict = deepcopy(self._model_module().state_dict())
            
        if last_loss is not None:
            self.bestLoss = last_loss            
            
        self.trained = True
        
    
    
    def save_state_dict(self, path, best = True, verbose = True):
        # In a distributed process group, only rank 0 writes shared files.
        if not self.is_main_process:
            return

        if best:
            state_dict = self.bestmodelStateDict
        else:
            state_dict = self._model_module().state_dict()

        torch.save(state_dict, path)
        if verbose:
            print("Model parameters saved to " + path )



    def trainModel(self, 
                   my_loss_function, train_dataloader,
                   earlyStop_dataloader = None,
                   max_epochs = 1000, patience = 50, 
                   saveModelEvery = torch.inf,
                   write_losses = False,                   
                   folder_temp = None, 
                   final_model_name = None,
                   verbose = True):
        
        """
        Trains the model using the provided training and early stopping dataloaders, with support for early stopping, 
        learning rate scheduling, and periodic model checkpointing.
        Args:
            my_loss (callable): Scalar batch-mean loss_function(y, output).
                Epoch metrics weight each batch by its sample count.
                Sum-reduced or variable-denominator masked losses must be
                normalized to a per-sample mean by the caller.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset. Must contain at least two tensors in each batch:
                - [0]: input
                - [-1]: target tensor (e.g. labels or ground truth)
                - Additional tensors can be passed to the model's forward pass, will use **forward_kwargs.
            earlyStop_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset used for early stopping. Format is the same as train_dataloader.
            If set to None, no early stopping is performed.
            max_epochs (int, optional): Maximum number of epochs to train. Defaults to 1000.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before stopping early. Defaults to 50.
            saveModelEvery (int or float, optional): Frequency (in epochs) to save model checkpoints. If set to torch.inf, disables periodic saving. Defaults to torch.inf.
            write_losses (bool, optional): Whether to save the training and validation loss arrays to disk after each epoch. Defaults to False.
            folder_temp (str, optional): Directory to save model checkpoints and loss arrays. Required if saveModelEvery is not torch.inf.
            final_model_name (str, optional): Path to save the final best model after training. If None, the model is not saved at the end. Defaults to None.
            verbose (bool, optional): Whether to print progress and status messages during training. Defaults to True.
        Raises:
            ValueError: If saveModelEvery is not torch.inf and folder_temp is not provided.
            ValueError: If both max_epochs and patience are set to torch.inf.
        Side Effects:
            - self.losses retains rank-local batch losses; self.lossesEpoch
              contains sample-weighted global training means. self.lossesTest
              contains sample-weighted validation means from rank 0's loader.
              DDP validation requires the full, unsharded validation loader.
            - Updates self.bestLoss and self.bestmodelStateDict with the best validation loss and corresponding model parameters.
            - Saves model checkpoints and loss arrays to disk if enabled.
            - Prints progress and status messages if verbose is True.
            - sets self.trained to True, indicating the model has been trained.            
        Returns:
            None
        """       
            
        self._validate_training_args(
            max_epochs,
            saveModelEvery,
            write_losses,
            folder_temp,
        )
                
        
        if self.trained:
            if verbose and self.is_main_process:
                print("Model has already been trained or loaded.")
                if self.losses == None:
                    print("Restarting losses and lossesTest tensors.")
                        
                print("Last recorded bestLoss: " + str(self.bestLoss))
            
            if self.bestLoss is None:
                self.bestLoss = torch.inf
        
            
        
        self.losses = torch.zeros((len(train_dataloader), max_epochs)) 
        self.losses[self.losses == 0] = torch.nan
        self.lossesEpoch = torch.full((max_epochs,), torch.nan, dtype=torch.float64)
        self.lossesTest = None
        
        if earlyStop_dataloader is None:
            pass
        else:
            self.lossesTest = torch.zeros(max_epochs)
            self.lossesTest[self.lossesTest == 0] = torch.nan


        
        if verbose and self.is_main_process:
            print("Starting training on " + str(self.device) + " with " + str(self.world_size) + " processes.")
            if self.ddp:
                print("DDP is enabled")
            
            
            
        epoch = 0
        patienceCounter = 0
        sumTime = 0
        elapsedTime = -1

        while (epoch < max_epochs) and (patienceCounter < patience):
            
            if verbose and self.is_main_process:
                print("\n-----------------")
                

                print("GPU " + str(self.device) + ", Epoch: " + str(epoch) + "/" + str(max_epochs) + " @ " + str(datetime.datetime.now()) + "(+" + str(elapsedTime) + "s)")
                print("Current learning rate is: " + str(self.optimizer.param_groups[0]['lr']))
                if epoch > 0:
                    print(f"Train Loss: {self.lossesEpoch[epoch-1]}")
                    if earlyStop_dataloader is not None:
                        print(f"Validation current loss: {self.lossesTest[epoch-1]} / Best loss : {self.bestLoss}")
                    else:
                        print(f"Current loss: {self.lossesEpoch[epoch-1]} / Best loss : {self.bestLoss}")
                        
                start = time.time()
                
            
            batch_i = 0
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            
            
            if self.ddp:
                train_dataloader.sampler.set_epoch(epoch)  # Ensure shuffling is consistent across epochs
            
            
            progress_bar = tqdm(
                train_dataloader,
                disable=not self.is_main_process,
            )
            for batch in progress_bar:
                if self.ddp:      
                    progress_bar.set_description(f"Running epoch {epoch} on {self.world_size} GPUs")
                else:
                    progress_bar.set_description(f"Running epoch {epoch} on device " + str(self.device))


                x = batch[0]
                y = batch[-1]
                extra = batch[1:-1]
                if len(extra) != 0:
                    forward_kwargs = { k: v for k, v in zip(self.additional_forward_args, extra) }
                else:
                    forward_kwargs = {}
                                    
                
                loss = self.train_iter(x = x,
                                       y = y, 
                                       loss_function = my_loss_function,
                                       **forward_kwargs)
                
                self.losses[batch_i, epoch] = loss
                epoch_loss_sum += loss * x.shape[0]
                epoch_sample_count += x.shape[0]
                                
                batch_i += 1
                


            self.lossesEpoch[epoch] = self._epoch_loss(epoch_loss_sum, epoch_sample_count)
            if earlyStop_dataloader is None:
                loss_to_account = self.lossesEpoch[epoch]
            else:
                self.__validate(my_loss_function, earlyStop_dataloader, epoch) # updates self.lossesTest[epoch]
                loss_to_account = self.lossesTest[epoch]
            
            
                
            if not (self.scheduler is None):
                self.scheduler_step(loss_to_account)



            if (loss_to_account < self.bestLoss):        
                patienceCounter = 0                
            else:
                patienceCounter += 1


            self.updateBestModelandLoss(loss_to_account) # already checks if loss is better than bestLoss and updates bestLoss


            if verbose and self.is_main_process:
                # print("Patience: " + str(patienceCounter) + "/" + str(patience))
                print("GPU " + str(self.device) + ", Patience: " + str(patienceCounter) + "/" + str(patience))
                elapsedTime = time.time() - start 
                sumTime += elapsedTime



            if ( (epoch + 1) % saveModelEvery ) == 0:
                self.save_state_dict( folder_temp + "/" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"),
                                    best = True, 
                                    verbose = verbose)

            if write_losses and self.is_main_process:
                np.save(folder_temp + "/losses.npy", self.losses.to("cpu").numpy())
                np.save(folder_temp + "/lossesEpoch.npy", self.lossesEpoch.numpy())
                
                if earlyStop_dataloader is not None:
                    np.save(folder_temp + "/lossesTest.npy", self.lossesTest.to("cpu").numpy()) 
                
                
                
            epoch += 1


        self.trained = True
        
        
        self._load_model_state_dict(self.bestmodelStateDict)

        self.losses = self.losses[:, :epoch ]
        self.lossesEpoch = self.lossesEpoch[:epoch]
        
        if earlyStop_dataloader is not None:        
            self.lossesTest = self.lossesTest[ :epoch ]
        
        
        
        if verbose and self.is_main_process:
            print("\n \n")
            print("Training finished after " + str(epoch) + " epochs.")
            print("Total training time: " + str(datetime.timedelta(seconds = sumTime)))
            print("Average time per epoch: " + str(sumTime / epoch))
            if earlyStop_dataloader is None:
                add_string = " (on training set): "
            else:    
                add_string = " (on validation set): "

            print("Best loss" + add_string  + str(self.bestLoss))
        
        
        
        if final_model_name is None:
            pass
        else:
            self.save_state_dict( final_model_name, best = True, verbose = verbose)




    def __validate(self, my_loss, earlyStop_dataloader, epoch):
        if len(earlyStop_dataloader) == 0:
            raise ValueError("earlyStop_dataloader is empty.")

        validation_loss = 0.0
        validation_sample_count = 0

        if not self.ddp or self.is_main_process:
            validation_model = self._model_module()

            for batch in earlyStop_dataloader:
                x2d_early_stop_b = batch[0]
                y_early_stop_b = batch[-1]
                extra = batch[1:-1]
                if len(extra) != 0:
                    forward_kwargs = {
                        k: v
                        for k, v in zip(self.additional_forward_args, extra)
                    }
                else:
                    forward_kwargs = {}

                batch_loss = self._test_iter_with_model(
                    validation_model,
                    x2d_early_stop_b,
                    y=y_early_stop_b,
                    loss_function=my_loss,
                    verbose=False,
                    **forward_kwargs,
                )

                batch_size = x2d_early_stop_b.shape[0]
                validation_loss += batch_loss * batch_size
                validation_sample_count += batch_size

            validation_loss /= validation_sample_count

        if self.ddp:
            validation_loss_tensor = torch.tensor(
                validation_loss,
                device=self.device,
                dtype=torch.float64,
            )
            dist.broadcast(validation_loss_tensor, src=0)
            validation_loss = validation_loss_tensor.item()

        self.lossesTest[epoch] = validation_loss




    def __validateDiffusion(self, my_loss, earlyStop_dataloader, noise_scheduler, epoch):
        if len(earlyStop_dataloader) == 0:
            raise ValueError("earlyStop_dataloader is empty.")

        validation_loss = 0.0
        validation_sample_count = 0

        if not self.ddp or self.is_main_process:
            validation_model = self._model_module()

            for clean_images, condition in earlyStop_dataloader:
                noise = torch.randn(clean_images.shape)
                timesteps = torch.randint(
                    low=0,
                    high=noise_scheduler.config['num_train_timesteps'],
                    size=(clean_images.shape[0],),
                    dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(
                    clean_images,
                    noise,
                    timesteps,
                )

                # Conditional path based on model input
                if "encoder_hidden_states" in self.additional_forward_args:
                    noisy_images_and_condition = noisy_images
                    forward_kwargs = {"encoder_hidden_states": condition}
                else:
                    noisy_images_and_condition = torch.cat(
                        (noisy_images, condition),
                        dim=1,
                    )
                    forward_kwargs = {}

                batch_loss = self._test_iter_with_model(
                    validation_model,
                    noisy_images_and_condition,
                    y=noise,
                    loss_function=my_loss,
                    verbose=False,
                    timestep=timesteps,
                    **forward_kwargs,
                )

                batch_size = clean_images.shape[0]
                validation_loss += batch_loss * batch_size
                validation_sample_count += batch_size

            validation_loss /= validation_sample_count

        if self.ddp:
            validation_loss_tensor = torch.tensor(
                validation_loss,
                device=self.device,
                dtype=torch.float64,
            )
            dist.broadcast(validation_loss_tensor, src=0)
            validation_loss = validation_loss_tensor.item()

        self.lossesTest[epoch] = validation_loss



     
        






    def trainDiffusionModel(self, 
                            train_dataloader,
                            noise_scheduler,
                            earlyStop_dataloader = None,
                            max_epochs = 1000, patience = 50, 
                            saveModelEvery = torch.inf,
                            write_losses = False,                   
                            folder_temp = None, 
                            final_model_name = None,
                            verbose = True,
                            my_loss_function = torch.nn.MSELoss(reduction = 'mean')):                    
        """
        Trains the model for diffusion tasks using the provided training dataloader and noise scheduler, 
        with support for early stopping, learning rate scheduling, and periodic model checkpointing.
        This model is meant to be used with conditional diffusion models,
            where the input is a noisy image and the target is the clean image.

        Differences from trainModel:
            - Designed for diffusion models, expects batches where the first channel is the target (clean images)
              and the remaining channels are conditioning information.
            - Requires a noise_scheduler to generate noisy images and timesteps.
            - Uses MSE loss between predicted and true noise. This is hardcoded since diffusion models work with MSE
            
        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset. Must contain two tensors in each batch:
                - [0]: target (clean images)
                - [1]: conditioning            
            my_loss_function (callable): Scalar batch-mean loss. Epoch metrics
                weight batches by sample count; sum-reduced or variable-denominator
                masked losses require caller normalization to a per-sample mean.
            max_epochs (int, optional): Maximum number of epochs to train. Defaults to 1000.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before stopping early. Defaults to 50.
            saveModelEvery (int or float, optional): Frequency (in epochs) to save model checkpoints. If set to torch.inf, disables periodic saving. Defaults to torch.inf.
            write_losses (bool, optional): Whether to save the training and validation loss arrays to disk after each epoch. Defaults to False.
            folder_temp (str, optional): Directory to save model checkpoints and loss arrays. Required if saveModelEvery is not torch.inf.
            final_model_name (str, optional): Path to save the final best model after training. If None, the model is not saved at the end. Defaults to None.
            verbose (bool, optional): Whether to print progress and status messages during training. Defaults to True.
        Raises:
            ValueError: If saveModelEvery is not torch.inf and folder_temp is not provided.
            ValueError: If both max_epochs and patience are set to torch.inf.
        Side Effects:
            - self.losses retains rank-local batch losses; self.lossesEpoch
              contains sample-weighted global training means. self.lossesTest
              contains sample-weighted validation means from rank 0's loader.
              DDP validation requires the full, unsharded validation loader.
            - Updates self.bestLoss and self.bestmodelStateDict with the best validation loss and corresponding model parameters.
            - Saves model checkpoints and loss arrays to disk if enabled.
            - Prints progress and status messages if verbose is True.
            - sets self.trained to True, indicating the model has been trained.
        Returns:
            None
        """                     
            
            
        self._validate_training_args(
            max_epochs,
            saveModelEvery,
            write_losses,
            folder_temp,
        )
        
        
        
        if self.trained:
            if verbose and self.is_main_process:
                print("Model has already been trained or loaded.")
                if self.losses == None:
                    print("Restarting losses and lossesTest tensors.")
                        
                print("Last recorded bestLoss: " + str(self.bestLoss))

            if self.bestLoss is None:
                self.bestLoss = torch.inf


            
        
        self.losses = torch.zeros((len(train_dataloader), max_epochs)) 
        self.losses[self.losses == 0] = torch.nan
        self.lossesEpoch = torch.full((max_epochs,), torch.nan, dtype=torch.float64)
        self.lossesTest = None
        
        if earlyStop_dataloader is None:
            pass
        else:
            self.lossesTest = torch.zeros(max_epochs)
            self.lossesTest[self.lossesTest == 0] = torch.nan
            
            
            
        if verbose and self.is_main_process:
            print("Starting training on " + str(self.device) + " with " + str(self.world_size) + " processes.")
            if self.ddp:
                print("DDP is enabled")
            
            
            
        epoch = 0
        patienceCounter = 0
        sumTime = 0
        elapsedTime = -1

        while (epoch < max_epochs) and (patienceCounter < patience):
            
            if verbose and self.is_main_process:
                print("\n-----------------")
                

                print("GPU " + str(self.device) + ", Epoch: " + str(epoch) + "/" + str(max_epochs) + " @ " + str(datetime.datetime.now()) + "(+" + str(elapsedTime) + "s)")
                print("Current learning rate is: " + str(self.optimizer.param_groups[0]['lr']))
                if epoch > 0:
                    if earlyStop_dataloader is not None:
                        print(f"Train Loss: {self.lossesEpoch[epoch-1]}")
                        print(f"Validation current loss: {self.lossesTest[epoch-1]} / Best loss : {self.bestLoss}")
                    else:
                        print(f"Train current loss: {self.lossesEpoch[epoch-1]} / Best loss: {self.bestLoss}")
                start = time.time()
                

            if self.ddp:
                train_dataloader.sampler.set_epoch(epoch)  # Ensure shuffling is consistent across epochs
            # Note from the warning in DistributedSampler
            # In distributed mode, calling the set_epoch method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
            
            
            batch_i = 0
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            progress_bar = tqdm(
                train_dataloader,
                disable=not self.is_main_process,
            )
            for clean_images, condition in progress_bar:          

                noise = torch.randn( clean_images.shape )

                timesteps = torch.randint( low = 0, high = noise_scheduler.config['num_train_timesteps'], 
                        size = (clean_images.shape[0], ), dtype=torch.long)
            
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # If model expects encoder_hidden_states (UNet2DConditionModel)
                # pass condition as a kwarg; otherwise, concatenate
                if "encoder_hidden_states" in self.additional_forward_args:
                    noisy_images_and_condition = noisy_images
                    forward_kwargs = {"encoder_hidden_states": condition}
                else:
                    noisy_images_and_condition = torch.cat((noisy_images, condition), dim = 1)
                    forward_kwargs = {}

                
                loss = self.train_iter(x = noisy_images_and_condition,
                                                              y = noise, 
                                                              timestep = timesteps,
                                                              loss_function = my_loss_function,
                                                              **forward_kwargs
                                                              )
                self.losses[batch_i, epoch] = loss
                epoch_loss_sum += loss * clean_images.shape[0]
                epoch_sample_count += clean_images.shape[0]

                if self.ddp:      
                    progress_bar.set_description(f"Running epoch {epoch} on {self.world_size} GPUs. Current Loss: {self.losses[batch_i, epoch]}")
                else:
                    progress_bar.set_description(f"Running epoch {epoch} on device " + str(self.device))

                                
                batch_i += 1
                                       

            
            self.lossesEpoch[epoch] = self._epoch_loss(epoch_loss_sum, epoch_sample_count)
            if earlyStop_dataloader is None:
                loss_to_account = self.lossesEpoch[epoch]
            else:
                self.__validateDiffusion(my_loss_function, earlyStop_dataloader, noise_scheduler, epoch) # updates self.lossesTest[epoch]
                loss_to_account = self.lossesTest[epoch]
            
            

            if not (self.scheduler is None):
                self.scheduler_step(loss_to_account)


            if (loss_to_account < self.bestLoss):        
                patienceCounter = 0                
            else:
                patienceCounter += 1


            self.updateBestModelandLoss(loss_to_account) # already checks if loss is better than bestLoss and updates bestLoss


            if verbose and self.is_main_process:
                # print("Patience: " + str(patienceCounter) + "/" + str(patience))
                print("GPU " + str(self.device) + ", Patience: " + str(patienceCounter) + "/" + str(patience))
                elapsedTime = time.time() - start 
                sumTime += elapsedTime



            if ( (epoch + 1) % saveModelEvery ) == 0:
                self.save_state_dict( folder_temp + "/" + str(epoch) + "_" + datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p"),
                                    best = True, 
                                    verbose = verbose)

            if write_losses and self.is_main_process:
                np.save(folder_temp + "/losses.npy", self.losses.to("cpu").numpy())
                np.save(folder_temp + "/lossesEpoch.npy", self.lossesEpoch.numpy())
                if earlyStop_dataloader is not None:
                    np.save(folder_temp + "/lossesTest.npy", self.lossesTest.numpy())
                
                
                
            epoch += 1


        self.trained = True
        self._load_model_state_dict(self.bestmodelStateDict)


        self.losses = self.losses[:, :epoch ]
        self.lossesEpoch = self.lossesEpoch[:epoch]
        if earlyStop_dataloader is not None:        
            self.lossesTest = self.lossesTest[ :epoch ]
                
        
        
        if verbose and self.is_main_process:
            print("\n \n")
            print("Training finished after " + str(epoch) + " epochs.")
            print("Total training time: " + str(datetime.timedelta(seconds = sumTime)))
            print("Average time per epoch: " + str(sumTime / epoch))
            
            if earlyStop_dataloader is not None:
                add_string = " (on validation set): "
            else:   
                add_string = " (on training set): "
            print("Best loss" + add_string  + str(self.bestLoss))
            

        
        if final_model_name is None:
            pass
        else:
            self.save_state_dict( final_model_name, best = True, verbose = verbose)





    def simulateDiffusion(self, dataloader, noise_scheduler, sub_batch_size = None, transformSimulation = None, file_name = None,
                          verbose = True, **step_kwargs):
        """
        Simulates the diffusion process over a dataset using a noise scheduler and saves or returns the results.
        Note: If launched on parallel, this method requires backend = "gloo", since it pushes the results to cpus
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing (sample, conditioning) pairs for simulation.
                Assumes sample is pregenerated noise (typically normally distributed noise)
                Will generate samples based on this
            noise_scheduler: An object that provides the diffusion timesteps and a `step` method to update samples.
            sub_batch_size (int, optional): If provided, the dataloader is split into sub-batches of this size for processing.
            transformSimulation (callable, optional): A function to apply to the simulation results before saving or returning.
            file_name (str, optional): If provided, the simulation results are saved to this file (as a NumPy .npy file).
                                        If None, the simulation results are returned as a NumPy array.
        Returns:
            np.ndarray or None: The simulated data as a NumPy array if `file_name` is None; otherwise, saves the data to disk and returns None.
        Notes:
            - If running in a distributed setting (`self.world_size > 1`), the file name is suffixed with the device rank.
            - The method assumes that `self.predict` and `noise_scheduler.step` are implemented and compatible with the data shapes.
            - The simulation is performed by iteratively applying the model's prediction and the noise scheduler's step for each timestep.
        """
        
        
        for thingys in dataloader: # to get sample shape
            sample = thingys[0]
            break
        
        if len(thingys) == 3:
            has_indexing = True
        else:
            has_indexing = False
            if self.world_size > 1:
                print("Warning: no index was found and this seems to be a distributed simulation. It will not be possible to remove potential duplicates and beware of the order.")
        
        
        if verbose and self.is_main_process:
            print("Simulating on " + str(self.world_size) + " devices")

        uses_subdataloader = False
        if sub_batch_size is not None:
            if sample.shape[0] > sub_batch_size:
                uses_subdataloader = True

                

        if uses_subdataloader:            
            progress_bar = dataloader  
        else:
            if verbose: 
                progress_bar = tqdm(
                    dataloader,
                    disable=not self.is_main_process,
                )
            else:
                progress_bar = dataloader


        
        use_encoder_conditioning = "encoder_hidden_states" in self.additional_forward_args



        simulation = torch.zeros((0, *sample.shape[1:]))
        if has_indexing:
            index = torch.zeros((0))                                     
        for thingys in progress_bar:
            

            
            sample = thingys[0] 
            conditioning = thingys[1]  
            if uses_subdataloader:
                if verbose:
                    sub_dataloader = tqdm(torch.utils.data.DataLoader( torch.utils.data.TensorDataset(sample, conditioning),
                                                                    batch_size = sub_batch_size,
                                                                    shuffle = False), 
                                        disable=not self.is_main_process
                                        )
                else:
                    sub_dataloader = torch.utils.data.DataLoader( torch.utils.data.TensorDataset(sample, conditioning),
                                                                batch_size = sub_batch_size,
                                                                shuffle = False)
            else:
                sub_dataloader = [(sample, conditioning)]
            
            
            training_mode = self.model.training
            self.model.eval()            
            with torch.inference_mode():                
                for sample, conditioning in sub_dataloader:    
                    batch_size, channels, height, width = sample.shape
                    conditioning = conditioning.to(self.device)
                    
                    
                    if not use_encoder_conditioning:
                        sample_and_condition = torch.empty( (batch_size, channels + conditioning.shape[1], height, width),
                                                            device=self.device, dtype=sample.dtype)
                        sample_and_condition[:, channels:, ... ].copy_(conditioning)


                    # rebuild a clean scheduler so step_index/model_outputs start fresh
                    noise_scheduler.set_timesteps(len(noise_scheduler.timesteps), device=self.device)

                    sample = sample * noise_scheduler.init_noise_sigma # initialize the latent with the scheduler sigma
                    sample = sample.to(self.device)    # makes sure it doesn't go back and forth between cpu and gpu, since self.predict will move it to self.device but then put it back to original device
                    
                    
                    
                    for timestep_ in noise_scheduler.timesteps:
                        #sample_and_condition = torch.cat((noise_scheduler.scale_model_input(sample, timestep_), conditioning), dim = 1) # concatenate the background prediction to the dataset
                        scaled_input = noise_scheduler.scale_model_input(sample, timestep_)
                        if use_encoder_conditioning:
                            residual = self.model(scaled_input, timestep=timestep_, encoder_hidden_states=conditioning)
                        else:
                            sample_and_condition[:, :channels, ... ].copy_(scaled_input)
                            residual = self.model(sample_and_condition, timestep=timestep_)

                        sample = noise_scheduler.step(model_output = residual, timestep = timestep_, sample = sample, **step_kwargs).prev_sample
                            
                    simulation = torch.concatenate((simulation, sample.cpu()), dim=0)

            self.model.train(mode=training_mode)

            if has_indexing:
                index = torch.concatenate((index, thingys[2]))
                
            
        
        if self.world_size > 1:
            # Gather the simulation results from all devices
            # Note: This assumes that the simulation is on the CPU, so necessitates backend = "gloo"
            simulations_on_all_devices = [torch.zeros(size = simulation.shape) for _ in range(self.world_size)] if self.is_main_process else None
            gather(simulation, simulations_on_all_devices, dst = 0)
            
            if has_indexing:
                indices_on_all_devices = [torch.zeros(size = index.shape) for _ in range(self.world_size)] if self.is_main_process else None
                gather(index, indices_on_all_devices, dst = 0)
        
        
        if self.is_main_process:
            if self.world_size > 1:            
                simulation = torch.cat(simulations_on_all_devices, dim=0)
                if has_indexing:
                    index = torch.cat(indices_on_all_devices, dim=0)
                    
                    # Sorts the simulation and removes duplicates:
                    simulation = simulation[np.unique(index, return_index = True)[1]]
        
        
            if transformSimulation is not None:    
                simulation = transformSimulation(simulation)                


            simulation = simulation.numpy()  # Convert to numpy array for saving
            
            if file_name is None:
                return simulation
            else:                
                np.save(file_name, simulation)


    """     
    # if Im gonna do this in parallel they should join the main process, the next is generated by copilot:
    def simulate_diffusion_parallel(self, dataloader, noise_scheduler, dataset_min, dataset_max, file_name):

        if not self.ddp:
            raise ValueError("This method is only available when using DDP.")
        
        # Each process will handle its own part of the data
        simulation = self.simulate_diffusion(dataloader, noise_scheduler, dataset_min, dataset_max, file_name)
        
        # Gather results from all processes
        gathered_simulations = [torch.zeros_like(simulation) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_simulations, simulation)
        
        # Concatenate results from all processes
        final_simulation = torch.cat(gathered_simulations, dim=0)
        
        # Save the final simulation
        np.save(file_name, final_simulation.cpu().numpy())
        
        return final_simulation.cpu().numpy()
    """
