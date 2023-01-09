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

1.1 Download data and models as of [here](Baselines/PartGlot/README.md) under `Data and pre-trained weights`
1.2 Add data files to `data/partglot` folder
1.3 Add and rename model under checkpoint to `models/partglot_pn_agnostic.ckpt`

2. Activate environment.

```bash
conda activate dl3d
```

3. Confirm Weights & Biases installation. More info: [click here](https://wandb.ai/quickstart/pytorch).

```bash
pip install wandb
wandb login
```

4. Enjoy ;)
