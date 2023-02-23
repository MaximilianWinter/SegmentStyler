import torch
import jstyleson

from src.submodels.special_layers import NumericsBackward
from src.models.original_model import Text2MeshOriginal


class Text2MeshExtended(Text2MeshOriginal):
    def __init__(self, args, data_dict):
        super().__init__(args, data_dict)

        self.previous_pred_rgb = torch.zeros_like(self.default_color)
        self.initial_pred_rgb = None
        if self.args.use_gt_masks:
            self.masks = data_dict["gt_masks"]
            # Gaussian weights not implemented yet for GT masks TODO
        else:
            self.masks = data_dict["masks"]
            self.gaussian_weights = data_dict["weights"]
            self.sigmas = data_dict["sigmas"]
            self.coms = data_dict["coms"]
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
        