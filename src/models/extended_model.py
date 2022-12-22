import torch
import jstyleson

from src.submodels.special_layers import NumericsBackward
from src.models.original_model import Text2MeshOriginal
from src.utils.utils import gaussian3D


class Text2MeshExtended(Text2MeshOriginal):
    def __init__(self, args, base_mesh):
        super().__init__(args, base_mesh)

        self.previous_pred_rgb = torch.zeros_like(self.default_color)
        self.initial_pred_rgb = None
        self.masks = self.load_masks()
        self.gaussian_weights = self.get_gaussian_weights_from_masks(self.masks)
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
        self.stylize_mesh(pred_rgb, pred_normal)

        # Rendering, Augmentations and CLIP encoding
        encoded_renders_dict, rendered_images = self.render_and_encode()

        color_reg = self.get_color_reg_terms(pred_rgb)

        self.previous_pred_rgb = pred_rgb.clone().detach()

        return {
            "encoded_renders": encoded_renders_dict,
            "rendered_images": rendered_images,
            "color_reg": color_reg,
        }

    def get_color_reg_terms(self, pred_rgb):
        """
        Extracts ground truth color regularizer.
        """
        used_tensor = None
        if self.args.use_previous_prediction:
            used_tensor = pred_rgb - self.previous_pred_rgb
        elif self.args.use_initial_prediction:
            used_tensor = pred_rgb - self.initial_pred_rgb
        else:
            used_tensor = pred_rgb

        color_reg = {}
        for prompt, mask in self.masks.items():
            color_reg[prompt] = torch.sum(
                used_tensor**2 * mask
            )  # penalizing term, to be added to the loss

        return color_reg

    def load_masks(self):
        with open(self.args.mask_path) as fp:
            mesh_metadata = jstyleson.load(fp)

        masks = {}
        for prompt in self.args.prompts:
            if "legs" in prompt:
                parts = ["leg_1", "leg_2", "leg_3", "leg_4"]
            else:
                parts = [
                    part
                    for part in mesh_metadata["mask_vertices"].keys()
                    if part in prompt
                ]
            mask = torch.ones_like(self.default_color)
            for part in parts:
                start, finish = mesh_metadata["mask_vertices"][part]
                mask[start:finish] = 0
                if self.args.noisy_masks:
                    n_tot = mask.shape[0]
                    n = finish - start
                    random_zeros = torch.randint(0, n_tot, (n // 5,))
                    random_ones = torch.randint(0, n_tot, (n // 5,))
                    mask[random_zeros] = 0
                    mask[random_ones] = 1
            masks[prompt] = mask

        return masks

    def get_gaussian_weights_from_masks(self, masks):
        """
        @param masks: dict with prompts as keys and masks as keys
        @returns: normalized_weights, dict of normalized gaussian weights
        """
        normalized_weights = {}
        sum_of_weights = None

        for prompt, mask in masks.items():
            inv_mask = 1 - mask
            part_vertices = self.base_mesh.vertices[inv_mask[:, 0].bool()].detach()

            COM = torch.mean(part_vertices, dim=0)
            Sigma = (
                (part_vertices - COM).T
                @ (part_vertices - COM)
                / (part_vertices.shape[0] - 1)
            )
            gauss_weight = gaussian3D(self.base_mesh.vertices, COM, Sigma)

            weight = torch.zeros_like(inv_mask)
            for i in range(weight.shape[1]):
                weight[:, i] = gauss_weight
            normalized_weights[prompt] = weight

            if sum_of_weights is None:
                sum_of_weights = weight.clone()
            else:
                sum_of_weights += weight

        for prompt in normalized_weights.keys():
            normalized_weights[prompt][sum_of_weights != 0] /= sum_of_weights[
                sum_of_weights != 0
            ]

        return normalized_weights
