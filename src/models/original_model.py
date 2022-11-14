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
        self.mlp.reset_weights()

        #### OTHER ####
        # Mesh
        self.base_mesh = base_mesh
        self.base_mesh_vertices = copy.deepcopy(base_mesh.vertices)

        # Prior color
        self.prior_color = torch.full(
            size=(self.base_mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

        # Default color
        self.default_color = torch.zeros(len(base_mesh.vertices), 3).to(device)
        self.default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
        # Background
        if self.args.background is None:
            self.background = None
        else:
            assert len(self.args.background) == 3
            self.background = torch.tensor(self.args.background).to(device)

    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)
        
        if not self.args.optimize_displacement:
            pred_normal[pred_normal != 0] = 0
            
        # Get stylized mesh
        self.base_mesh.face_attributes = self.prior_color + kal.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0),
            self.base_mesh.faces)

        self.base_mesh.vertices = self.base_mesh_vertices + self.base_mesh.vertex_normals * pred_normal

        MeshNormalizer(self.base_mesh)()

        # Rendering
        rendered_images, elev, azim = self.renderer.render_front_views(self.base_mesh, num_views=self.args.n_views,
                                                                show=self.args.show,
                                                                center_azim=self.args.frontview_center[0],
                                                                center_elev=self.args.frontview_center[1],
                                                                std=self.args.frontview_std,
                                                                return_views=True,
                                                                background=self.background)
        geo_renders = None
        if self.args.geoloss:
            self.base_mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(self.default_color.unsqueeze(0),
                                                                                    self.base_mesh.faces)
            
            geo_renders, elev, azim = self.renderer.render_front_views(self.base_mesh, num_views=self.args.n_views,
                                                                    show=self.args.show,
                                                                    center_azim=self.args.frontview_center[0],
                                                                    center_elev=self.args.frontview_center[1],
                                                                    std=self.args.frontview_std,
                                                                    return_views=True,
                                                                    background=self.background)

        # Augmentations and CLIP encoding
        encoded_renders_dict = self.clip_with_augs.get_encoded_renders(rendered_images, geo_renders)

        mask = torch.ones_like(pred_rgb)
        mask[20618:29364] = 0 # those are the seat vertices
        #mask[0:12912] = 0 # those are the back vertices
        #mask[12912:20618] = 1 # this is one leg
        #mask[29364:37060] = 1 # this is one leg
        #mask[37060:44691] = 1 # this is one leg
        #mask[44691:52299] = 1 # this is one leg

        color_reg = torch.sum(pred_rgb**2*mask) # penalizing term, to be added to the loss

        return {"encoded_renders": encoded_renders_dict, "rendered_images": rendered_images, "color_reg": color_reg}