import torch
import torch.nn as nn
import kaolin as kal
import copy

from src.submodels.clip_with_augs import CLIPWithAugs
from src.submodels.render import Renderer
from src.submodels.neural_style_field import NeuralStyleField
from src.utils.render import get_render_resolution
from src.utils.Normalization import MeshNormalizer
from src.utils.utils import device


class Text2MeshOriginal(nn.Module):
    def __init__(self, args, data_dict):
        super().__init__()
        self.args = args

        #### MODELS ####
        # CLIP
        self.clip_with_augs = CLIPWithAugs(args)
        res = get_render_resolution(args.clipmodel)

        # Renderer
        self.renderer = Renderer(dim=(res, res))

        # MLP
        self.input_dim = 6 if self.args.input_normals else 3
        if self.args.only_z:
            self.input_dim = 1
        self.mlp = NeuralStyleField(args, input_dim=self.input_dim).to(device)
        self.mlp.reset_weights()

        #### OTHER ####
        # Mesh
        base_mesh = data_dict["mesh"]
        self.base_mesh = base_mesh
        self.base_mesh_vertices = copy.deepcopy(base_mesh.vertices)

        # Prior color
        self.prior_color = torch.full(
            size=(self.base_mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device
        )

        # Default color
        self.default_color = torch.zeros(len(base_mesh.vertices), 3).to(device)
        self.default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)

        # Background
        if self.args.background is None:
            self.background = None
        else:
            assert len(self.args.background) == 3
            self.background = torch.tensor(self.args.background).to(device)

        self.stop_loop = False
        
    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)

        # Get stylized mesh
        self.stylize_mesh(pred_rgb, pred_normal)

        # Rendering, Augmentations and CLIP encoding
        encoded_renders_dict, rendered_images = self.render_and_encode()

        return {
            "encoded_renders": encoded_renders_dict,
            "rendered_images": rendered_images,
        }

    def stylize_mesh(self, pred_rgb, pred_normal):
        self.base_mesh.face_attributes = (
            self.prior_color
            + kal.ops.mesh.index_vertices_by_faces(
                pred_rgb.unsqueeze(0), self.base_mesh.faces
            )
        )

        self.base_mesh.vertex_colors = pred_rgb + 0.5

        if self.args.optimize_displacement:
            self.base_mesh.vertices = (
                self.base_mesh_vertices + self.base_mesh.vertex_normals * pred_normal
            )
        else:
            self.base_mesh.vertices = self.base_mesh_vertices

        MeshNormalizer(self.base_mesh)()

    def render_and_encode(self, return_views=False, views=None):
        # Rendering
        if views is not None:
            rendered_images, elev, azim = self.renderer.render_given_front_views(
                self.base_mesh,
                views["elev"],
                views["azim"],
                show=self.args.show,
                return_views=True,
                background=self.background,
            )
        else:
            rendered_images, elev, azim = self.renderer.render_front_views(
                self.base_mesh,
                num_views=self.args.n_views,
                show=self.args.show,
                center_azim=self.args.frontview_center[0],
                center_elev=self.args.frontview_center[1],
                std=self.args.frontview_std,
                return_views=True,
                background=self.background,
            )
        geo_renders = geo_elev = geo_azim = None
        if self.args.geoloss:
            self.base_mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
                self.default_color.unsqueeze(0), self.base_mesh.faces
            )

            if views is not None:
                geo_renders, geo_elev, geo_azim = self.renderer.render_given_front_views(
                self.base_mesh,
                views["elev"],
                views["azim"],
                show=self.args.show,
                return_views=True,
                background=self.background,
            )
            else:
                geo_renders, geo_elev, geo_azim = self.renderer.render_front_views(
                    self.base_mesh,
                    num_views=self.args.n_views,
                    show=self.args.show,
                    center_azim=self.args.frontview_center[0],
                    center_elev=self.args.frontview_center[1],
                    std=self.args.frontview_std,
                    return_views=True,
                    background=self.background,
                )

        # Augmentations and CLIP encoding
        encoded_renders_dict = self.clip_with_augs.get_encoded_renders(
            rendered_images, geo_renders
        )

        if return_views:
            views = {"elev": elev, "azim": azim, "geo_elev": geo_elev, "geo_azim": geo_azim}
            return encoded_renders_dict, rendered_images, views
        else:
            return encoded_renders_dict, rendered_images
