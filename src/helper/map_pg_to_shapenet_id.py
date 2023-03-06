"""
Helper script for mapping PartGlot's internal sample
ids to ShapeNet ids.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.shapenet import ShapeNetPoints
from src.partglot.utils.partglot_bspnet_preprocess import (
    normalize_pointcloud,
    rotate_pointcloud,
)
from src.partglot.utils.processing import vstack2dim
from src.helper.preprocessing import get_pc_distances

data = ShapeNetPoints()
points = []
for i in tqdm(range(len(data))):
    points.append(data[i]["points"])

h5_data = h5py.File("/mnt/hdd/PartGlotData/shapenet_partseg_chair_bsp.h5")
segs_data = h5_data["data"][:].astype(np.float32)

prepros_sn_pc = [
    normalize_pointcloud(rotate_pointcloud(vstack2dim(pc)))["pc"] for pc in points
]

pg_idx_to_shapenet_idx = {}
all_dists = []
for i, src_pc in enumerate(tqdm(segs_data)):
    n_src = normalize_pointcloud(vstack2dim(src_pc))["pc"]
    pc_dists = get_pc_distances(n_src, prepros_sn_pc)
    closest_pc = min(pc_dists, key=lambda t: t[1])
    print(closest_pc)
    pg_idx_to_shapenet_idx[i] = closest_pc[0]
    all_dists.append(pc_dists)

p = Path("/mnt/hdd/PartGlotData/data_mapping_chair_bsp.txt")
with open(p, "w") as fp:
    for pg_idx, shapenet_idx in pg_idx_to_shapenet_idx.items():
        synset_id, item_id = data[shapenet_idx]["name"].split("-")
        fp.write(f"{pg_idx}/{synset_id}/{item_id}\n")
