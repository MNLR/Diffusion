"""Training objectives shared by training and validation."""

import torch


class DiffusionObjective:
    """The existing conditional diffusion objective: predict sampled noise.

    This class prepares model inputs and targets and evaluates the supplied
    loss. Model execution, device transfers, optimizer updates and distributed
    communication remain the responsibility of the training engine.

    Args:
        noise_scheduler: Existing scheduler providing ``add_noise`` and
            ``config['num_train_timesteps']``.
        loss_function: Scalar batch-mean callable ``loss_function(target,
            prediction)``. Model.trainDiffusionModel supplies MSE by default.
        use_encoder_conditioning: Pass conditioning as ``encoder_hidden_states``
            when True; otherwise concatenate it with noisy images by channel.

    This objective always targets epsilon, irrespective of scheduler prediction
    settings. It preserves the existing random draws and conditioning behaviour;
    it does not introduce other diffusion objectives or prediction types.
    """

    def __init__(self, noise_scheduler, loss_function,  #fixme Think I'm gonna make loss function fixed in this case
                 use_encoder_conditioning=False):
        self.noise_scheduler = noise_scheduler
        self.loss_function = loss_function
        self.use_encoder_conditioning = use_encoder_conditioning

    def prepare_batch(self, clean_images, conditioning):
        """Return ``(inputs, noise_target, forward_kwargs)`` for one batch.

        Keep the original shape-based random draws, their order and the global
        RNG. In particular, do not replace randn with randn_like or move the
        draws to another device as part of this extraction.
        """
        noise = torch.randn(clean_images.shape)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config['num_train_timesteps'],
            size=(clean_images.shape[0],),
            dtype=torch.long,
        )
        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps,
        )

        forward_kwargs = {"timestep": timesteps}
        if self.use_encoder_conditioning:
            inputs = noisy_images
            forward_kwargs["encoder_hidden_states"] = conditioning
        else:
            inputs = torch.cat((noisy_images, conditioning), dim=1)

        return inputs, noise, forward_kwargs

    def compute_loss(self, target, prediction):
        """Evaluate the supplied loss, retaining target-first argument order."""
        return self.loss_function(target, prediction)
