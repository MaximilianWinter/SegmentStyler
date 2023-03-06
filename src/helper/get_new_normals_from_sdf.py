"""
Helper script for remeshing ShapeNet shapes and fixing their normals.
Used as pre-processing.
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_voxels # recommended to install this in a separate environment
import trimesh
import skimage
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.helper.paths import GLOBAL_DATA_PATH

def normalize_shapenet(mesh):
    diag = mesh.bounds[1] - mesh.bounds[0]
    mesh.vertices = (mesh.vertices - mesh.centroid)/np.linalg.norm(diag)

def save_obj(mesh, path="./test.obj"):
    exp_string = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=False, include_texture=False, return_texture=False, write_texture=False)
    with open(path, "w") as fp:
            fp.write(exp_string)


mapping_path = GLOBAL_DATA_PATH.joinpath("PartGlotData/data_mapping_chair_bsp.txt")
dataset_path = GLOBAL_DATA_PATH.joinpath("PartGlotData")
shapenet_path = GLOBAL_DATA_PATH.joinpath("ShapeNetCore.v2")

items = mapping_path.read_text().splitlines()
for item in tqdm(items):
    _, synset_id, item_id = item.split("/")
    shapenet_mesh_path = shapenet_path.joinpath(f"{synset_id}/{item_id}/models/model_normalized.obj")
    mesh_path = dataset_path.joinpath(f"{synset_id}/{item_id}/mesh.obj")
    Path(mesh_path.parent).mkdir(parents=True, exist_ok=True)

    shapenet_mesh = trimesh.load(shapenet_mesh_path, force="mesh")
    print("Get voxels...")
    voxels = mesh_to_voxels(shapenet_mesh, 128, pad=True, sign_method="depth")

    print("Remesh...")
    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=-normals) # for some reason the normals have to be flipped here to be correct
    normalize_shapenet(new_mesh)
    print("Save...")
    save_obj(new_mesh, path=mesh_path)