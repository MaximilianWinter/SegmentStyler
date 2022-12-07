import pathlib as pl
import os

root_path = pl.Path(os.path.abspath(__file__)).parents[1]

NOTEBOOKS_PATH = root_path / 'notebooks'
BASELINES_PATH = root_path / 'Baselines'
DATA_PATH = root_path / 'data'