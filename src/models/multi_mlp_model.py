import torch
import torch.nn as nn
import kaolin as kal

from src.models.extended_model import Text2MeshExtended
from src.submodels.neural_style_field import NeuralStyleField
from src.submodels.special_layers import MaskBackward
from src.utils.utils import device, gaussian3D


class Text2MeshMultiMLP(Text2MeshExtended):
    def __init__(self, args, data_dict):
        super().__init__(args, data_dict)

        self.mlp = None
        mlp_dict = {}
        for prompt in args.prompts:
            mlp = NeuralStyleField(args, input_dim=self.input_dim).to(device)
            mlp.reset_weights()

            mlp_dict[prompt] = mlp

        self.mlp = nn.ModuleDict(mlp_dict)

        self.mask_backward = MaskBackward.apply

    def forward(self, vertices):
        # Prop. through MLPs
        pred_rgb, pred_normal = self.prop_through_mlps(vertices)

        # Rendering, Augmentations and CLIP encoding per prompt
        (
            encoded_renders_dict_per_prompt,
            rendered_images_per_prompt,
            color_reg,
        ) = self.render_augment_encode(vertices, pred_rgb, pred_normal)

        return {
            "encoded_renders": encoded_renders_dict_per_prompt,
            "rendered_images": rendered_images_per_prompt,
            "color_reg": color_reg,
        }

    def biased_render_and_encode(
        self, center_point, distance, covariances, principal_axes, num_vertices
    ):
        # Rendering
        rendered_images, _, _ = self.renderer.render_sampled_views_along_principal_axes(
            self.base_mesh,
            covariances,
            principal_axes,
            num_vertices,
            num_views=self.args.n_views,
            show=self.args.show,
            center_azim=self.args.frontview_center[0],
            center_elev=self.args.frontview_center[1],
            center_point=center_point,
            distance=distance,
            std=self.args.frontview_std,
            return_views=True,
            background=self.background,
        )
        geo_renders = None
        if self.args.geoloss:
            self.base_mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
                self.default_color.unsqueeze(0), self.base_mesh.faces
            )

            geo_renders, _, _ = self.renderer.render_sampled_views_along_principal_axes(
                self.base_mesh,
                covariances,
                principal_axes,
                num_vertices,
                num_views=self.args.n_views,
                show=self.args.show,
                center_azim=self.args.frontview_center[0],
                center_elev=self.args.frontview_center[1],
                center_point=center_point,
                distance=distance,
                std=self.args.frontview_std,
                return_views=True,
                background=self.background,
            )

        # Augmentations and CLIP encoding
        encoded_renders_dict = self.clip_with_augs.get_encoded_renders(
            rendered_images, geo_renders
        )

        return encoded_renders_dict, rendered_images

    def prop_through_mlps(self, vertices):
        # Prop. through MLPs
        pred_rgb = None
        pred_normal = None
        for prompt, mlp in self.mlp.items():
            pred_rgb_per_prompt, pred_normal_per_prompt = mlp(vertices)
            inv_mask = 1 - self.masks[prompt]
            if self.args.gaussian_blending:
                weight = self.gaussian_weights[prompt]
            else:
                weight = inv_mask
            pred_rgb_masked = pred_rgb_per_prompt * weight
            pred_normal_masked = pred_normal_per_prompt * weight

            if pred_rgb is not None:
                pred_rgb += pred_rgb_masked
            else:
                pred_rgb = pred_rgb_masked

            if pred_normal is not None:
                pred_normal += pred_normal_masked
            else:
                pred_normal = pred_normal_masked

        if self.args.round_renderer_gradients:
            pred_rgb = self.num_backward(pred_rgb)
            pred_normal = self.num_backward(pred_normal)

        if self.initial_pred_rgb is None:
            self.initial_pred_rgb = pred_rgb.clone().detach()

        return pred_rgb, pred_normal

    def render_augment_encode(self, vertices, pred_rgb, pred_normal):
        # Rendering, Augmentations and CLIP encoding per prompt
        encoded_renders_dict_per_prompt = {}
        rendered_images_per_prompt = None
        
        for i, prompt in enumerate(self.args.prompts):
            inv_mask = 1 - self.masks[prompt]
            if self.args.gaussian_blending:
                weight = self.gaussian_weights[prompt]
            else:
                weight = inv_mask
            # Get stylized mesh
            if self.args.do_backward_masking:
                pred_rgb_masked = self.mask_backward(pred_rgb, weight)
                pred_normal_masked = self.mask_backward(pred_normal, weight)

                self.stylize_mesh(pred_rgb_masked, pred_normal_masked)
            else:
                self.stylize_mesh(pred_rgb, pred_normal)

            if self.args.biased_views:
                if inv_mask.bool().any():
                    center_point = torch.mean(
                        vertices[inv_mask[:, 0].bool()], dim=0
                    )  # we use the part's COM
                    distance = (
                        torch.sum((vertices[inv_mask[:, 0].bool()] - center_point) ** 2, dim=1)
                        .sqrt()
                        .max()
                        * 2
                    )  # we use 2-times the part's expansion
                    (U, S, V) = torch.pca_lowrank(vertices[inv_mask[:, 0].bool()])
                    m = vertices[inv_mask[:, 0].bool()].shape[0]
                    encoded_renders_dict, rendered_images = self.biased_render_and_encode(
                        center_point, distance, S, V, m
                    )
                else:
                    print("Could not bias view as part is not present in mask.")
                    encoded_renders_dict, rendered_images = self.render_and_encode()

            else:
                if i == 0:
                    encoded_renders_dict, rendered_images, views = self.render_and_encode(return_views=True)
                else:
                    encoded_renders_dict, rendered_images = self.render_and_encode(views=views)
            
            encoded_renders_dict_per_prompt[prompt] = encoded_renders_dict
            if rendered_images_per_prompt is None:
                rendered_images_per_prompt = rendered_images
            else:
                rendered_images_per_prompt = torch.cat(
                    [rendered_images_per_prompt, rendered_images], dim=0
                )

        color_reg = self.get_color_reg_terms(pred_rgb)

        self.previous_pred_rgb = pred_rgb.clone().detach()

        return encoded_renders_dict_per_prompt, rendered_images_per_prompt, color_reg
