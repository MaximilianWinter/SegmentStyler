import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin as kal
import jstyleson
import copy
from src.models.fix_numerics import NumericsBackward

from src.submodels.clip_with_augs import CLIPWithAugs
from src.submodels.render import Renderer
from src.submodels.neural_style_field import NeuralStyleField
from src.utils.render import get_render_resolution
from src.utils.Normalization import MeshNormalizer
from src.utils.utils import device


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
        self.mlp = NeuralStyleField(self.args.sigma, self.args.depth, self.args.width, self.args.encoding, self.args.colordepth, self.args.normdepth,
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

        self.previous_pred_rgb = torch.zeros_like(self.default_color)

        self.initial_pred_rgb = None

        self.masks = self.load_masks()

        self.num_backward = NumericsBackward.apply

    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)
        if self.args.round_renderer_gradients:
            pred_rgb = self.num_backward(pred_rgb)
            pred_normal = self.num_backward(pred_normal)

        if self.initial_pred_rgb is None:
            self.initial_pred_rgb = pred_rgb.clone().detach()
            
        # Get stylized mesh
        self.base_mesh.face_attributes = self.prior_color + kal.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0),
            self.base_mesh.faces)

        self.base_mesh.vertices = self.base_mesh_vertices + self.base_mesh.vertex_normals * pred_normal

        if self.args.optimize_displacement:
            self.base_mesh.vertices = self.base_mesh_vertices + self.base_mesh.vertex_normals * pred_normal
        else:
            self.base_mesh.vertices = self.base_mesh_vertices

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

        if self.args.use_previous_prediction:
            color_reg = self.get_color_reg(pred_rgb-self.previous_pred_rgb)
        elif self.args.use_initial_prediction:
            color_reg = self.get_color_reg(pred_rgb-self.initial_pred_rgb)
        else:
            color_reg = self.get_color_reg(pred_rgb)
        self.previous_pred_rgb = pred_rgb.clone().detach()

        return {"encoded_renders": encoded_renders_dict, "rendered_images": rendered_images, "color_reg": color_reg}
    
        
    def get_color_reg(self, pred_rgb):
        """
        Extracts ground truth color regularizer.
        """        
        color_reg = {}
        for prompt, mask in self.masks.items():
            color_reg[prompt] = torch.sum(pred_rgb**2*mask) # penalizing term, to be added to the loss
        
        return color_reg
        
    def load_masks(self):
        with open(self.args.mask_path) as fp:
            mesh_metadata = jstyleson.load(fp)

        masks = {}
        for prompt in self.args.prompts:
            parts = [part for part in mesh_metadata["mask_vertices"].keys() if part in prompt]
            mask = torch.ones_like(self.default_color)
            for part in parts:
                start, finish = mesh_metadata["mask_vertices"][part]
                mask[start:finish] = 0
            masks[prompt] = mask
            
        return masks