<div align="center">
  <h1>OLMo-core</h1>
</div>
<p align="center">
  <a href="XXX">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="XXX">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/OLMo.svg">
  </a>
  <a href="XXX">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
</p>

`OLMo-core` is a repository for training and using OLMo3, AI2's state-of-the-art open language model. It is designed by scientists, for scientists.

## Installation
Create or activate a Python virtual environment with a Python version ≥ 3.10, then install [PyTorch](https://pytorch.org/).

Next, we recommend installing `OLMo-core` from source:
```bash
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```

## Pretraining

Official training scripts for released models can be found in `src/scripts/official/`. These scripts are meant to be launched with `torchrun`.

To get started, we'll use the script `src/examples/llm/train.py` to launch a short language model pretraining run with a small transformer (271M params) on a subset of c4. This will only take a few minutes on as little as 2 NVIDIA 40GB A100s.

### Defining a config
Let's understand how the key components and hyperparameters of the run are defined. Near the top of the script you’ll find the config dataclass:

```python
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder.
    Mainly used when you want to fine-tune from a pretrained model."""
    load_trainer_state: bool = False
    """Whether to load the trainer state (including data loader state) when loading from `load_path`.
    This only makes sense when trainer state is available in the checkpoint and you're resuming
    on the same dataset."""
```

Using a config class is not necessary, but has several benefits:
1. It gives us a good way to keep track of all the hyperparameters of each experiment. Since the config inherits from OLMo-core’s [Config](https://olmo-core.readthedocs.io/en/latest/config.html#olmo_core.config.Config) baseclass, it comes with useful methods to serialize it to JSON which, for example, could be uploaded to Weights & Biases or saved to the run’s checkpoint directory.
2. It gives us a command-line argument parser that enables us to override fields in the config at runtime. For instance, we can add `--data_loader.prefetch_factor=4` to the command to update the `prefetch_factor` field within the `data_loader` part of the config.

To validate that our overrides are applied correctly, we can print the experimental config using the `--dry-run` flag. We also pass in the name of the run, which is the only positional argument expected by the script.

```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run
```

And now let's try it again while overriding a few config options:
```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run \
  --data_loader.prefetch_factor=4 \
  --trainer.callbacks.wandb.enabled=true
```
### Launching the run
Now that we know how to change settings on the fly, we're ready to launch the run. For the first run, we'll use overrides to turn off some features that we'd normally want on.
- `--trainer.callbacks.lm_evaluator.enabled=false` disables the in-loop perplexity evaluator.
- `--trainer.callbacks.downstream_evaluator.enabled=false` disables the in-loop downstream task evaluator.
- `--trainer.no_checkpoints` disables checkpointing.
- `--trainer.hard_stop='{value: 100, unit: steps}'` terminates the training at step 100.

Assuming you have two GPUs available, the command would be
```
torchrun --nproc-per-node=2 src/examples/llm/train.py \
  tutorial-run-01 \
  --save-folder=/tmp/tutorial-run-01 \
  --work-dir=/tmp/dataset-cache \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 100, unit: steps}'
```
### Tokenizing new data
