from src.models.extended_model import Text2MeshExtended
from src.submodels.special_layers import MaskBackward


class Text2MeshBackwardMask(Text2MeshExtended):
    def __init__(self, args, base_mesh):
        super().__init__(args, base_mesh)
        print("Initializing Text2MeshBackwardMask...")
        self.mask_backward = MaskBackward.apply

    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)

        # Do backward masking
        for mask_per_prompt in self.masks.values():
            if mask is not None:
                mask += mask_per_prompt
            else:
                mask = mask_per_prompt

        pred_rgb = self.mask_backward(
            pred_rgb, 1 - mask
        )  # inverting the mask, because the mask is originally defined for the penalizing loss

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
