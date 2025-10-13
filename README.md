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
```
git clone https://github.com/allenai/OLMo-core.git
cd OLMo-core
pip install -e .[all]
```

## Pretraining

Official training scripts for released models can be found in `src/scripts/official/`. These scripts are meant to be launched with `torchrun`.

We'll start by launching a short language model pretraining run with a small transformer (271M params) on a subset of c4. This will only take a few minutes on as little as 2 NVIDIA 40GB A100s.

### Defining a config
Before launching the training run, let's look at how the key components and hyperparameters of the run are defined. The script expects one position argument, the name of the run. 

The script also includes a dry-run mode that prints the experimental config so that you can validate that your overrides are applied correctly. Let's try this right now by passing in the `--dry-run` flag.
```
python src/examples/llm/train.py tutorial-run-01 --dry-run
```

And now let's try it again while overriding a few config options:
```
python src/examples/llm/train.py tutorial-run-01 --dry-run \
  --data_loader.prefetch_factor=4 \
  --trainer.callbacks.wandb.enabled=true
```
### Launching the run
Now that we know how to change settings on the fly, we're ready to launch the run. Assuming that you have 2 GPUs available, the command would be
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
