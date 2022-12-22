import os
import os.path as osp
from multiprocessing import Pool

import h5py
import numpy as np
import trimesh

from src.helper.paths import BASELINES_PATH, LOCAL_DATA_PATH
from src.partglot.utils.processing import random_sample_array
from src.partglot.utils.simple_utils import unpickle_data

"""
Note that you need to make pairs of BSP-Net meshes and pointclouds have the same index order.
Fill in the paths below:
"""
GAME_DATA_PATH = LOCAL_DATA_PATH / "partglot_email/shapenet_chairs_only_in_game_10000.h5" # partglot game data path.
BSP_DATA_DIR = BASELINES_PATH / "BSP-NET-pytorch/samples/bsp_ae_out" # dir storing BSP-Net output meshes.
PC_DATA_PATH = BASELINES_PATH / "BSP-NET-pytorch/data/data_per_category/03001627_chair/03001627_vox256_img_test.hdf5" # path to the attached point cloud data.
OUTPUT_DIR = LOCAL_DATA_PATH / "preprocess_bspnet" # dir to save outputs from this preprocessing code.

def rotate_pointcloud(pc):
    # To align pointclouds to bspnet outputs
    rot = np.array([[0,0,-1], [0,1,0], [-1,0,0.]])
    rot_pc = pc @ rot.transpose()
    return rot_pc

def normalize_pointcloud(pc, boundary="cube"):
    # unit cube normalization
    if boundary == "cube":
        maxv, minv = np.max(pc, 0), np.min(pc,0)
        offset = minv
        pc = pc - offset
        scale = np.sqrt(np.sum((maxv-  minv) ** 2))
        pc = pc / scale
    elif boundary == "sphere":
        offset = np.mean(pc, 0)
        pc = pc - offset
        scale = np.max(np.sqrt(np.sum(pc ** 2, 1)))
        pc = pc / scale

    return dict(pc=pc, offset=offset, scale=scale)

def padding_pointcloud(pc, max_num_points=512, seed=63):
    # To make # of points in each super-segs. be 512.
    np.random.seed(seed)
    N = pc.shape[0]
    k = max_num_points // N + 1
    dup_pc = np.concatenate([pc for _ in range(k)])
    orig_pc, dup_pc = dup_pc[:N], dup_pc[N:]

    rid = np.arange(dup_pc.shape[0])
    np.random.shuffle(rid)
    dup_pc = dup_pc[rid]

    pad_pc = np.concatenate([orig_pc, dup_pc], 0)[:max_num_points]
    return pad_pc

def load_pointcloud(idx, num_points=None, res=64):
    h5_data = h5py.File(PC_DATA_PATH)
    pc_data = h5_data[f'points_{res}']
    mask = h5_data[f'values_{res}'][idx].astype(bool)
    mask = (np.ones((mask.shape[0], 3)) * mask).astype(bool)
    pc = pc_data[idx][mask].reshape((-1, 3))[:num_points]
    pc = rotate_pointcloud(pc)
    pc_label = None
    return pc, pc_label

def scene_as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values() if g.faces.shape[1] == 3))
    else:
        mesh = scene_or_mesh

    return mesh

def normalize_mesh(mesh: trimesh.Trimesh):
    # unit cube normalization
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    maxv, minv = np.max(v, 0), np.min(v, 0)
    offset = minv
    v = v - offset
    scale = np.sqrt(np.sum((maxv - minv) ** 2))
    v = v / scale
    normed_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return dict(mesh=normed_mesh, offset=offset, scale=scale)

def normalize_scene(scene: trimesh.Scene):
    mesh_merged = scene_as_mesh(scene)

    out = normalize_mesh(mesh_merged)
    offset = out["offset"]
    scale = out["scale"]

    submesh_normalized_list = []
    for i, submesh in enumerate(list(scene.geometry.values())):
        v, f = np.array(submesh.vertices), np.array(submesh.faces)
        v = v - offset
        v = v / scale
        submesh_normalized_list.append(trimesh.Trimesh(v, f))
        
    return trimesh.Scene(submesh_normalized_list)

def load_bsp_mesh(idx):
    mesh_path = osp.join(BSP_DATA_DIR, f"{idx}_bsp.obj")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)

    num_segments = len(mesh.geometry)
    new_mesh = []
    for i, g in enumerate(list(mesh.geometry.values())):
        new_mesh.append(g)
    new_mesh.reverse()
    new_mesh = trimesh.Scene(new_mesh)
    return new_mesh

def load_pointcloud_bsp_mesh_pair(idx, num_points=None, res=64):
    pc, pc_label = load_pointcloud(idx, num_points, res)
    mesh = load_bsp_mesh(idx)

    return dict(pc=pc, mesh=mesh, pc_label=pc_label)

def save_scene(scene: trimesh.Scene, filename):
    vertices = []
    with open(filename, "w") as fout:
        fout.write("mtllib default.mtl\n")
        for i, m in enumerate(list(scene.geometry.values())):
            v = m.vertices
            f = m.faces
            vbias = len(vertices) + 1
            vertices = vertices + list(v)

            for ii in range(len(v)):
                fout.write(f"v {v[ii][0]} {v[ii][1]} {v[ii][2]}\n")
            for ii in range(len(f)):
                fout.write("f ")
                for jj in range(len(f[ii])):
                    fout.write(f"{f[ii][jj]+vbias} ")
                fout.write("\n")

def measure_signed_distance(mesh: trimesh.Scene, pc):
    out = normalize_pointcloud(pc)
    pc = out["pc"]

    mesh = normalize_scene(mesh)
    
    signed_distance = np.zeros((len(mesh.geometry), pc.shape[0]))

    for i, g in enumerate(list(mesh.geometry.values())):
        sd = trimesh.proximity.signed_distance(g, pc)
        signed_distance[i] = sd

    signed_distance = signed_distance.transpose()

    return signed_distance # [num_points, num_meshes]


def save_signed_distance(idx, filename):
    item = load_pointcloud_bsp_mesh_pair(idx)
    pc = item["pc"]
    mesh = item["mesh"]
    sd = measure_signed_distance(mesh, pc)
    np.savetxt(filename, sd)

    
def assign_label_from_pc_to_primitive(pc_label, sd, num_labels=4):
    n_point, n_mesh = sd.shape
    point_membership = np.argmax(sd, 1)
    voting_box = np.zeros((n_mesh, num_labels))
    
    for i in range(n_mesh):
        near_point_idx = np.where(point_membership == i)
        l = pc_label[near_point_idx]
        for j in range(num_labels):
            voting_box[i,j] += (l==j).sum()
    mesh_label = np.argmax(voting_box, 1)

    return mesh_label

def assign_label_from_primitive_to_pc(mesh_label, sd):
    point_membership = np.argmax(sd, 1)
    assign_func = lambda x : mesh_label[x]
    pc_label = assign_func(point_membership)
    
    return pc_label

def reassign_pc_label_from(pc_label, sd, num_labels=4):
    """
    merge assing_label_from_pc_to_primitive() and assign_label_from_primitive_to_pc()
    """
    mesh_label = assign_label_from_pc_to_primitive(pc_label, sd, num_labels)
    return assign_label_from_primitive_to_pc(mesh_label, sd)

def get_bsp_attrb(idx, num_points=None, res=64):
    out = load_pointcloud_bsp_mesh_pair(idx, num_points, res)
    mesh, pc = out["mesh"], out["pc"]

    signed_distance = measure_signed_distance(mesh, pc) # [n_point, n_segs]
    
    point_membership = np.argmax(signed_distance, 1)
    
    return dict(mesh=mesh, pc=pc, signed_distance=signed_distance, point_membership=point_membership)
 
def convert_supersegs_to_pointclouds(idx, num_points=None, remove_rare=10, max_num_points=512, max_num_segs=50, res=64, export=True):
    bsp_attrb = get_bsp_attrb(idx, num_points, res)
    
    mesh, pc, point_membership, signed_distance = bsp_attrb['mesh'], bsp_attrb['pc'], bsp_attrb['point_membership'], bsp_attrb['signed_distance']

    pc_in_segs = []
    new_mesh = []
    new_sd = []
    num_points_list = []
    for i, g in enumerate(list(mesh.geometry.values())):
        points = pc[point_membership == i]
        num_points_list.append(points.shape[0])
        if len(points) < remove_rare:
            continue
        points = padding_pointcloud(points, max_num_points)
        pc_in_segs.append(points)
        new_mesh.append(g)
        sd = padding_pointcloud(signed_distance[point_membership == i], max_num_points)
        new_sd.append(sd)
    new_mesh = trimesh.Scene(new_mesh)
    new_signed_distance = np.stack(new_sd, 1)

    assert len(new_mesh.geometry) == len(pc_in_segs) == new_signed_distance.shape[1]
    assert len(pc_in_segs) < max_num_segs

    res = max_num_segs - len(pc_in_segs)
    dummy = np.zeros((max_num_points, 3))
    mask = [1] * len(pc_in_segs)

    for _ in range(res):
        pc_in_segs.append(dummy)
        mask.append(0)

    mask = np.array(mask)
    
    pc_in_segs = np.stack(pc_in_segs, 0) #[n_segs, max_num_points, 3]
    num_segs = int((mask==1).sum())
    pc_in_segs2 = pc_in_segs[:num_segs].reshape(-1, 3)
    pc_in_segs2 = normalize_pointcloud(pc_in_segs2, "sphere")
    pc_in_segs2 = pc_in_segs2['pc'].reshape(num_segs, max_num_points, 3)
    pc_in_segs[:num_segs] = pc_in_segs2[:num_segs]
    
    mask = np.array(mask)
    
    if export:
        """ Re-save data after removing tiny super-segments. """
        mesh_filename = f"{OUTPUT_DIR}/{idx}_new.obj"
        save_scene(new_mesh, mesh_filename)

        sd_filename = f"{OUTPUT_DIR}/{idx}_sd.txt"
        save_signed_distance(idx, sd_filename)

        pc_in_segs_filename = f"{OUTPUT_DIR}/{idx}_pc_in_segs.npy"
        np.save(pc_in_segs_filename, pc_in_segs)

        mask_filename = f"{OUTPUT_DIR}/{idx}_mask.npy"
        np.save(mask_filename, mask)

    pc2sup_segs = np.arange(mask.sum()).astype(np.int32)
    non_zero_pc_in_segs = pc_in_segs[mask.astype(bool)]
    
    return dict(mesh=new_mesh, pc_in_segs=pc_in_segs, mask=mask, non_zero_pc_in_segs=non_zero_pc_in_segs, sd=new_signed_distance, pc2sup_segs=pc2sup_segs)

def resample_ssegs(sseg_pc, sseg_size=512, normalize=False, with_replacement=True, min_sseg_size=10):
    resampled_ssegs = []
    for pc in sseg_pc:
        pc_array = np.array(pc)
        if pc_array.shape[0] < min_sseg_size:
            continue
        resampled_pc = random_sample_array(pc_array, size=sseg_size, with_replacement=with_replacement)
        if normalize:
            resampled_pc = normalize_pointcloud(resampled_pc, boundary="sphere")['pc']
        resampled_ssegs.append(resampled_pc)
    return np.array(resampled_ssegs)

def cluster_sseg_pointcloud(pointcloud, point_membership):
    seg_pc = {}
    for pc, pm in zip(pointcloud, point_membership):
        if pm in seg_pc:
            seg_pc[pm].append(pc)
        else:
            seg_pc[pm] = [pc]
            
    return seg_pc

def convert_supersegs_to_pointclouds_simple(idx, sseg_size=512, normalize=False, num_points=None, res=64, with_replacement=True):
    attr = get_bsp_attrb(idx, num_points=num_points, res=res)
    sseg_pc = cluster_sseg_pointcloud(attr['pc'], attr['point_membership'])
    resampled_seg_pc = resample_ssegs(sseg_pc.values(), normalize=normalize, sseg_size=sseg_size, with_replacement=with_replacement)
    return resampled_seg_pc
