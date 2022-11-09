from pathlib import Path

import pytorch_lightning as pl
from torchvision import transforms
import torch
from helpers import get_background, check_previous_run

from utils import device
import os


class MeshDataModule(pl.LightningDataModule):
    pass
