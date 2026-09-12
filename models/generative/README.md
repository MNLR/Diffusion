# Generative components

This directory contains components extracted from `models/DDPModel.py` in small,
behaviour-preserving steps. The first implemented class is `DiffusionObjective`
in `objectives.py`; the other modules remain documentation-only placeholders.

Existing experiment scripts continue to use `from models.DDPModel import Model`.
No script or configuration changes are required.

## Current objective class

`Model.trainDiffusionModel(...)` constructs one `DiffusionObjective` from its
existing scheduler, loss function and conditioning mode. Training and validation
share that instance:

```python
objective = DiffusionObjective(
    noise_scheduler=noise_scheduler,
    loss_function=my_loss_function,
    use_encoder_conditioning=use_encoder_conditioning,
)
inputs, target, forward_kwargs = objective.prepare_batch(clean_images, condition)
# Existing train/test step methods run the network and call:
loss = objective.compute_loss(target, prediction)
```

`prepare_batch` samples Gaussian noise and integer timesteps, calls the existing
scheduler's `add_noise`, and either concatenates conditioning channels or passes
`encoder_hidden_states`. The target remains the sampled noise. `compute_loss`
calls the supplied loss with the existing target-first argument order; the
training method still defaults to mean-reduced MSE.

The extraction retains the original global RNG, random-draw order, dtype/device
defaults and conditioning selection. Existing train/test step methods still own
device transfers, network execution, gradient handling and model train/eval mode.
DDP validation still uses the unwrapped network on rank zero and broadcasts its
loss. This is an extraction of the current epsilon objective, not support for
additional prediction types or schedulers.

## Module responsibilities

| Module | Intended responsibility |
| --- | --- |
| `processes.py` | Construct states from clean data, noise and time; define noise schedules and time conventions. |
| `objectives.py` | Implemented: existing epsilon-prediction `DiffusionObjective`, shared by training and validation. |
| `adapters.py` | Explicit conditioning, network-specific time encoding and normalization of network output formats. |
| `samplers.py` | Evolve a batch using a compatible trained method and sampling configuration; return tensors in transformed data space. |

The process, adapter and sampler responsibilities are boundaries for later work,
not a finalized API. No base classes or registries are introduced at this stage.

## Responsibilities outside this package

- The training engine owns DDP, optimizer updates, learning-rate scheduling,
  validation orchestration, early stopping and checkpoint coordination.
- Experiment runners own data loading, fitted transforms, configurations, seeds,
  dataset batching, distributed result collection, file output and diagnostics.
- Checkpoint utilities own serialization of weights and, when implemented,
  resumable training state.

Those responsibilities remain in their existing locations. Potential future
extractions into `models/trainer.py`, `functions/experiment.py` and
`functions/checkpoints.py` are deliberately deferred until there is code to move.

## Working with TestingHall

`Diffusion_TestingHall/models` links to `Diffusion/models`, so TestingHall sees
this package automatically. Maintain shared implementation in Diffusion and keep
experiment runs and test instrumentation in TestingHall.

`Diffusion_TestingHall/TEST_diffusion_objective.py` compares this extraction with
the pre-refactor `DDPModel.py` at Git commit
`c844af3008381d12977471ad28aa3f0fe3a39580`, which matched the working source when
the extraction began. From TestingHall, run:

```bash
python TEST_diffusion_objective.py
python TEST_diffusion_objective.py --ddp
```

The script uses synthetic data and a tiny network with the existing DDPM
scheduler. It compares inputs, targets, forward arguments, losses, gradients,
best and saved weights, optimizer/scheduler state, model mode and CPU RNG state.
It covers both conditioning paths, default MSE and an asymmetric custom loss,
uneven batches, and training with or without validation. The optional two-process
CPU/Gloo check also verifies rank-zero validation and agreement between ranks.
These checks do not exercise CUDA/NCCL or establish climate-model skill.

## Next step, when requested

Consider extracting batch sampling while keeping the existing public methods as
entry points. Verify any extraction against the preceding implementation with
controlled inputs and random state, including DDP consistency checks.

Keep bug fixes and methodological changes separate from that extraction. This
refactor adds no flow matching, alternate schedulers, prediction targets, seed
changes, transform changes or training optimizations.
