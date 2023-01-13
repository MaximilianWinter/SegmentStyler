import torch

from src.models.multi_mlp_model import Text2MeshMultiMLP
from src.submodels.gauss_estimator import GaussEstimator
from src.utils.utils import device, gaussian3D

class Text2MeshLearnedBlending(Text2MeshMultiMLP):
    def __init__(self, args, base_mesh):
        super().__init__(args, base_mesh)

        self.gauss_estimator = GaussEstimator(
            args, input_dim=self.input_dim, output_dim=3, n_prompts=len(args.prompts)
        ).to(device)

    def forward(self, vertices):
        com_input = torch.cat([t.unsqueeze(0) for t in self.coms.values()])
        mu_displacements = self.gauss_estimator(com_input)
        normalized_weights = self.get_normalized_weights(vertices, mu_displacements)

        # Prop. through MLPs
        pred_rgb, pred_normal = self.prop_through_mlps(vertices, normalized_weights)

        # Rendering, Augmentations and CLIP encoding per prompt
        (
            encoded_renders_dict_per_prompt,
            rendered_images_per_prompt,
            color_reg,
        ) = self.render_augment_encode(
            vertices, pred_rgb, pred_normal, normalized_weights
        )

        return {
            "encoded_renders": encoded_renders_dict_per_prompt,
            "rendered_images": rendered_images_per_prompt,
            "color_reg": color_reg,
        }

    def get_normalized_weights(self, vertices, mu_displacements):
        sum_of_weights = None
        normalized_weights = {}

        for i, prompt in enumerate(self.masks.keys()):
            inv_mask = 1 - self.masks[prompt]
            gauss_weight = gaussian3D(
                vertices, self.coms[prompt] - mu_displacements[i], self.sigmas[prompt]
            )
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
