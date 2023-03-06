import torch

from src.models.extended_model import Text2MeshExtended
from src.submodels.special_layers import MaskBackward


class Text2MeshBackwardMasking(Text2MeshExtended):
    def __init__(self, args, data_dict):
        """
        This model uses backward masking with a single MLP (neural styler).
        @param args: Namespace, defining configuration
        @param data_dict: dictionary, containing all relevant data, see corresponding dataset classes for details
        """
        super().__init__(args, data_dict)
        self.mask_backward = MaskBackward.apply

    def forward(self, vertices):
        """
        Forward pass.
        @param vertices: torch.tensor, shape (N, 3)
        @returns: dict, containing renderings and their encodings (in CLIP space)
        """
        pred_rgb, pred_normal = self.mlp(vertices)
        (
            encoded_renders_dict_per_prompt,
            rendered_images_per_prompt,
        ) = self.render_augment_encode(pred_rgb, pred_normal)

        return {
            "encoded_renders": encoded_renders_dict_per_prompt,
            "rendered_images": rendered_images_per_prompt,
        }

    def render_augment_encode(self, pred_rgb, pred_normal):
        """
        Wrapper function for stylizing, rendering, applying augmentations and encoding.
        Includes backward masking.
        @param pred_rgb, torch.tensor, shape (N, 3)
        @param pred_normal, torch.tensor, shape (N, 3)
        """
        encoded_renders_dict_per_prompt = {}
        rendered_images_per_prompt = None

        for i, prompt in enumerate(self.args.prompts):
            inv_mask = 1 - self.masks[prompt]
            if self.args.gaussian_blending:
                weight = self.gaussian_weights[prompt]
            else:
                weight = inv_mask

            if self.args.do_backward_masking:
                pred_rgb_masked = self.mask_backward(pred_rgb, weight)
                pred_normal_masked = self.mask_backward(pred_normal, weight)

                self.stylize_mesh(pred_rgb_masked, pred_normal_masked)
            else:
                self.stylize_mesh(pred_rgb, pred_normal)

            if i == 0:
                encoded_renders_dict, rendered_images, views = self.render_and_encode(
                    return_views=True
                )
            else:
                encoded_renders_dict, rendered_images = self.render_and_encode(
                    views=views
                )

            encoded_renders_dict_per_prompt[prompt] = encoded_renders_dict
            if rendered_images_per_prompt is None:
                rendered_images_per_prompt = rendered_images
            else:
                rendered_images_per_prompt = torch.cat(
                    [rendered_images_per_prompt, rendered_images], dim=0
                )
                
        return encoded_renders_dict_per_prompt, rendered_images_per_prompt
