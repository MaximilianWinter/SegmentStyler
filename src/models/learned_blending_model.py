import torch

from src.models.multi_mlp_model import Text2MeshMultiMLP
from src.submodels.gauss_estimator import GaussEstimator
from src.utils.utils import device, gaussian3D

class Text2MeshLearnedBlending(Text2MeshMultiMLP):
    def __init__(self, args, data_dict):
        """
        Module used for estimating the Gaussian Blending's center points using CLIP loss.
        This is not used in our current implementation.
        @param args: Namespace, defining configuration
        @param data_dict: dictionary, containing all relevant data, see corresponding dataset classes for details
        """
        super().__init__(args, data_dict)

        self.gauss_estimator = GaussEstimator(
            args, input_dim=self.input_dim, output_dim=3, n_prompts=len(args.prompts)
        ).to(device)
        self.gauss_estimator.reset_weights()

    def forward(self, vertices):
        unsqueezed_coms = []
        for com_list in self.coms.values():
            unsqueezed_coms.extend([com.unsqueeze(0) for com in com_list])
        com_input = torch.cat(unsqueezed_coms)
        mu_displacements = self.gauss_estimator(com_input)
        mu_displacements_dict = {}
        lower_idx = 0
        upper_idx = 0
        for prompt, com_list in self.coms.items():
            upper_idx += len(com_list)
            mu_displacements_dict[prompt] = mu_displacements[lower_idx:upper_idx]
            lower_idx = upper_idx
        normalized_weights = self.get_normalized_weights(vertices, mu_displacements_dict)
        self.gaussian_weights = normalized_weights
        # Prop. through MLPs
        pred_rgb, pred_normal = self.prop_through_mlps(vertices)

        # Rendering, Augmentations and CLIP encoding per prompt
        (
            encoded_renders_dict_per_prompt,
            rendered_images_per_prompt,
            color_reg,
        ) = self.render_augment_encode(
            vertices, pred_rgb, pred_normal
        )

        return {
            "encoded_renders": encoded_renders_dict_per_prompt,
            "rendered_images": rendered_images_per_prompt,
            "color_reg": color_reg,
        }

    def get_normalized_weights(self, vertices, mu_displacements):
        sum_of_weights = None
        normalized_weights = {}

        for prompt, mask in self.masks.items():
            inv_mask = 1 - mask
            gauss_weights = []
            for com, mu, sigma in zip(self.coms[prompt], mu_displacements[prompt], self.sigmas[prompt]):
                gauss_weights.append(gaussian3D(vertices, com - mu, sigma).unsqueeze(0))
            gauss_weight = torch.sum(torch.cat(gauss_weights, dim=0), dim=0)

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
    