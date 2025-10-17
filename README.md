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
The `ExperimentConfig` dataclass is defined near the top of the script. 

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

To override any fields in the config at runtime, we can simply add them as command-line options. For instance, adding `--data_loader.prefetch_factor=4` will update the `prefetch_factor` field within the `data_loader` part of the config. The script also specifies a subset of config options that we expect to be modified especially often as command-line arguments, namely the `--save-folder`, `work-dir`, and `--sequence-length`.

To validate that our overrides are applied correctly, we can print the experimental config using the `--dry-run` flag. Note that the single positional argument expected by the script is the name of the run.

```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run
```

And now let's try it again while overriding a few config options:
```bash
python src/examples/llm/train.py tutorial-run-01 --dry-run \
  --data_loader.prefetch_factor=4 \
  --trainer.callbacks.wandb.enabled=true
```

Finally, we can change the model architecture itself via the `--model-factory` argument. The options for this argument are the various classmethods of [TransformerConfig](https://olmo-core.readthedocs.io/en/latest/nn/transformer.html#olmo_core.nn.transformer.TransformerConfig), which define preset model configurations. Alternatively, you can also replace the following lines of the script with a particular `TransformerConfig` instance. For example, to hard-code in an OLMo2 1B model, you can replace these lines:

```
    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    model_config = factory(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )
```

with

```
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size()
    )
```

To specify a new model config, we recommend creating a new classmethod under `TransformerConfig`. Keep in mind that as you change the model size and architecture you’ll likely want to adjust hyperparameters and performance settings such as the learning rate and micro-batch size (`--train_module.rank_microbatch_size`).

### Launching the run
Now that we know how to change settings on the fly, we're ready to launch the run. For the first run, we'll use overrides to disable the in-loop perplexity evaluator, in-loop downstream task evaluator, checkpoint, and terminate the training at step 100. If you have two GPUs available, the command would be
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

### Finetuning pretrained models

This script can be used for finetuning pretrained models as well. To tell `Trainer` to load pretrained weights at the beginning of the run, use the `--load-path` option. You may also need to convert your model into a format that the `Trainer` expects. See this [HF conversion guide](You need to convert the pretrained weights into a format that the Trainer expects. See this HF conversion guide for an example of converting weights from HuggingFace into the right format.) for an example of converting weights from HuggingFace into the right format.

# Tokenizing new data
