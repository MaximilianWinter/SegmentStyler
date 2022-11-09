import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin as kal
import copy

from submodels.clip_with_augs import CLIPWithAugs
from submodels.render import Renderer
from submodels.neural_style_field import NeuralStyleField
from utils.render import get_render_resolution
from utils.Normalization import MeshNormalizer
from utils.utils import device


class Text2MeshOriginal(nn.Module):

    def __init__(self, args, base_mesh):
        super().__init__()
        self.args = args

        #### MODELS ####
        # CLIP
        self.clip_with_augs = CLIPWithAugs(args)
        self.res = get_render_resolution(args.clipmodel)

        # Renderer
        self.renderer = Renderer(dim=(self.res, self.res))

        # MLP
        self.input_dim = 6 if self.args.input_normals else 3
        if self.args.only_z:
            self.input_dim = 1
        self.mlp = NeuralStyleField(self.args.sigma, self.args.depth, self.args.width, 'gaussian', self.args.colordepth, self.args.normdepth,
                                    self.args.normratio, self.args.clamp, self.args.normclamp, niter=self.args.n_iter,
                                    progressive_encoding=self.args.pe, input_dim=self.input_dim, exclude=self.args.exclude).to(device)

        #### OTHER ####
        # Mesh
        self.base_mesh = base_mesh

        # Prior color
        self.prior_color = torch.full(
            size=(self.base_mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

        # Default color
        self.default_color = torch.zeros(len(base_mesh.vertices), 3).to(device)
        self.default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
        # Background
        self.background = self.get_background()

    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)

        # Get stylized mesh
        base_mesh_copy = copy.deepcopy(self.base_mesh)
        base_mesh_copy.face_attributes = self.prior_color + kal.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0),
            base_mesh_copy.faces)

        base_mesh_copy.vertices = vertices + base_mesh_copy.vertex_normals * pred_normal

        MeshNormalizer(base_mesh_copy)()

        # Rendering
        rendered_images, elev, azim = self.renderer.render_front_views(base_mesh_copy, num_views=self.args.n_views,
                                                                show=self.args.show,
                                                                center_azim=self.args.frontview_center[0],
                                                                center_elev=self.args.frontview_center[1],
                                                                std=self.args.frontview_std,
                                                                return_views=True,
                                                                background=self.background)

        
        base_mesh_copy.face_attributes = kal.ops.mesh.index_vertices_by_faces(self.default_color.unsqueeze(0),
                                                                                   base_mesh_copy.faces)
        
        geo_renders, elev, azim = self.renderer.render_front_views(base_mesh_copy, num_views=self.args.n_views,
                                                                show=self.args.show,
                                                                center_azim=self.args.frontview_center[0],
                                                                center_elev=self.args.frontview_center[1],
                                                                std=self.args.frontview_std,
                                                                return_views=True,
                                                                background=self.background)

        # Augmentations and CLIP encoding
        encoded_renders_dict = self.clip_with_augs.get_encoded_renders(rendered_images, geo_renders)

        return encoded_renders_dict