import trimesh
import pymeshlab
import numpy as np
from pathlib import Path
import json
import torch
from chamferdist import ChamferDistance
from tqdm import tqdm


def remesh_per_part(obj_path, save_path, remesh_iterations=6):
    """
    Preprocessing function for remeshing each part of a mesh. In order to preserve the part information,
    we store an additional .json file with face and vertex offsets for the individual parts.

    :param obj_path: str, specifying the obj file to be read
    :param save_path: str, specifying the file path (.obj) for saving the remeshed mesh, an additonal .json file with offsets is
    saved at the same location
    :remesh_iterations: int, number of iterations, passed to meshing_isotropic_explicit_remeshing method of pymeshlab
    """
    with open(obj_path) as fp:
        mesh_dict = trimesh.exchange.obj.load_obj(
            fp, skip_materials=True, maintain_order=False, group_material=False
        )
    ms = pymeshlab.MeshSet()

    for value in mesh_dict["geometry"].values():
        ml_mesh = pymeshlab.Mesh(
            value["vertices"], value["faces"], v_normals_matrix=value["vertex_normals"]
        )
        ms.add_mesh(ml_mesh)
        ms.meshing_isotropic_explicit_remeshing(iterations=remesh_iterations)

    vertex_list = []
    vertex_normals_list = []
    face_list = []
    vertex_index_offsets = [0]
    face_index_offsets = [0]

    vertex_index_offset = 0
    face_index_offset = 0
    for submesh in ms:
        vertex_list.append(submesh.vertex_matrix())
        vertex_normals_list.append(submesh.vertex_normal_matrix())
        face_list.append(submesh.face_matrix() + vertex_index_offset)

        vertex_index_offset += len(submesh.vertex_matrix())
        face_index_offset += len(submesh.face_matrix())

        vertex_index_offsets.append(vertex_index_offset)
        face_index_offsets.append(face_index_offset)

    vertex_array = np.concatenate(vertex_list)
    vertex_normals_array = np.concatenate(vertex_normals_list)
    face_array = np.concatenate(face_list)

    export_string = trimesh.exchange.obj.export_obj(
        trimesh.Trimesh(vertex_array, face_array, vertex_normals=vertex_normals_array)
    )
    save_path = Path(save_path)
    with open(save_path, "w") as fp:
        fp.write(export_string)

    offset_path = save_path.parent.joinpath(save_path.stem + "_offsets.json")
    with open(offset_path, "w") as fp:
        json.dump(
            {
                "vertex_index_offsets": vertex_index_offsets,
                "face_index_offsets": face_index_offsets,
            },
            fp,
        )



chamferDist = ChamferDistance()

def chamfer_dist(src_pc:np.array, dst_pc:np.array, bidirectional=True, boundary="cude"):
    """noramlize pointcloud before feeding here"""
    assert src_pc.shape[0] != 1 and dst_pc.shape[0] != 1
    src = torch.from_numpy(src_pc).cuda().unsqueeze(0).float()
    dst = torch.from_numpy(dst_pc).cuda().unsqueeze(0).float()
    dist_forward = chamferDist(src, dst, bidirectional=bidirectional)
    return dist_forward.detach().cpu().item()


def get_pc_distances(src_pc, pc_list):
    out = []
    for i,dst_pc in tqdm(enumerate(pc_list)):
        cd = chamfer_dist(src_pc, dst_pc)
        out.append((i, cd))
    return out