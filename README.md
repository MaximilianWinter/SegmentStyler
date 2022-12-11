# TUM DL3D Practical
## Language guided mesh editing / generation models with local details

### Purpose of this repo

> Explore language guided models for mesh editing and generation with the goal of achieving local details. 

### How to run

1. Run `setup.sh`.

```bash
sh ./setup.sh
```

1. Setup PartGlot 

2.1 Download input data as of [here](Baselines/PartGlot/README.md) under `Data and pre-trained weights`
2.2 Pickle PartGlot Model

```
cd ./Baselines/PartGlot
pip install -e .
python pickle_model.py
cd ../..
```

1. Activate environment.

```bash
conda activate dl3d
```

4. Confirm Weights & Biases installation. More info: [click here](https://wandb.ai/quickstart/pytorch).

```bash
pip install wandb
wandb login
```

4. Enjoy ;)
