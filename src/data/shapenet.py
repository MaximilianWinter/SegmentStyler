from pathlib import Path
import torch
import numpy as np

from src.data.mesh import Mesh
from src.helper.paths import GLOBAL_DATA_PATH

class ShapeNet(torch.utils.data.Dataset):
    dataset_path = GLOBAL_DATA_PATH.joinpath("ShapeNetCore.v2")

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self.items = (
            ShapeNet.dataset_path.joinpath("data_list.txt")
            .read_text()
            .splitlines()
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        synset_id, item_id = self.items[index].split("/")
        mesh = ShapeNet.get_mesh(synset_id, item_id)
        
        return {
            "name": f"{synset_id}-{item_id}",
            "mesh": mesh
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        raise NotImplementedError

    @staticmethod
    def get_mesh(synset_id, item_id):
        mesh = Mesh(
            str(
                ShapeNet.dataset_path.joinpath(
                    f"{synset_id}/{item_id}/models/model_normalized.obj"
                )
            )
        )
        return mesh


class ShapeNetPoints(torch.utils.data.Dataset):
    dataset_path = GLOBAL_DATA_PATH.joinpath("shapenetcore_partanno_segmentation_benchmark_v0")

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self.items = (
            ShapeNetPoints.dataset_path.joinpath("data_list.txt")
            .read_text()
            .splitlines()
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        synset_id, item_id = self.items[index].split("/")
        points = ShapeNetPoints.get_points(synset_id, item_id)
        
        return {
            "name": f"{synset_id}-{item_id}",
            "points": points
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        raise NotImplementedError

    @staticmethod
    def get_points(synset_id, item_id):
        points = np.loadtxt(ShapeNetPoints.dataset_path.joinpath(f"{synset_id}/points/{item_id}.pts"))
        return points

