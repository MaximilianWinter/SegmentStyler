import kaolin as kal
import trimesh
import torch
from src.utils.utils import device
from src.utils.mesh import get_texture_map_from_color, get_face_attributes_from_color, standardize_mesh, normalize_mesh, add_vertices
import copy
import numpy as np
import PIL

class MeshContainer:
    def __init__(self, vertices, faces, vertex_normals, face_normals):
        self.vertices = torch.tensor(vertices).float()
        self.faces = torch.tensor(faces).long()
        self.vertex_normals = torch.tensor(vertex_normals).float()
        self.face_normals = torch.tensor(face_normals).float()

class Mesh():
    def __init__(self,path_or_trimesh,color=torch.tensor([0.0,0.0,1.0]), use_trimesh=False):
        if isinstance(path_or_trimesh, str):
            obj_path = path_or_trimesh
            if ".obj" in obj_path:
                if use_trimesh:
                    with open(obj_path) as fp:
                        mesh_dict = trimesh.exchange.obj.load_obj(fp, include_color=False, include_texture=False)
                    tri_mesh = trimesh.Trimesh(**mesh_dict)
                    mesh = MeshContainer(tri_mesh.vertices, tri_mesh.faces, tri_mesh.vertex_normals, tri_mesh.face_normals)
                else:
                    mesh = kal.io.obj.import_mesh(obj_path, with_normals=True)
            elif ".off" in obj_path:
                mesh = kal.io.off.import_mesh(obj_path)
            else:
                raise ValueError(f"{obj_path} extension not implemented in mesh reader.")
        elif isinstance(path_or_trimesh, trimesh.Trimesh):
            tri_mesh = path_or_trimesh
            mesh = MeshContainer(tri_mesh.vertices, tri_mesh.faces, tri_mesh.vertex_normals, tri_mesh.face_normals)
        else:
            raise TypeError("path_or_trimesh must be of type str or trimesh.Trimesh.")
        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.vertex_normals = None
        self.face_normals = None
        self.texture_map = None
        self.face_uvs = None
        self.vertex_colors = None
        if (isinstance(path_or_trimesh, str) and ".obj" in path_or_trimesh) or isinstance(path_or_trimesh, trimesh.Trimesh):
            # if mesh.uvs.numel() > 0:
            #     uvs = mesh.uvs.unsqueeze(0).to(device)
            #     face_uvs_idx = mesh.face_uvs_idx.to(device)
            #     self.face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
            if mesh.vertex_normals is not None:
                self.vertex_normals = mesh.vertex_normals.to(device).float()

                # Normalize
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            if mesh.face_normals is not None:
                self.face_normals = mesh.face_normals.to(device).float()

                # Normalize
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

        self.set_mesh_color(color)

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return standardize_mesh(mesh)

    def normalize_mesh(self,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return normalize_mesh(mesh)

    def update_vertex(self,verts,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self,color):
        self.texture_map = get_texture_map_from_color(self,color)
        self.face_attributes = get_face_attributes_from_color(self,color)

    def set_image_texture(self,texture_map,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map,str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)


        mesh.texture_map = texture_map
        return mesh

    def divide(self,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_face_uvs = add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.face_uvs = new_face_uvs
        return mesh

    def export(self, file, color=None):
        with open(file, "w+") as f:
            for vi, v in enumerate(self.vertices):
                if color is None:
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                if self.vertex_normals is not None:
                    f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))

