from src.models.extended_model import Text2MeshExtended
from src.submodels.neural_style_field import NeuralStyleField
from src.utils.utils import device


class Text2MeshMultiMLP(Text2MeshExtended):

    def __init__(self, args, base_mesh):
        super().__init__(args, base_mesh)

        self.mlp = None
        self.mlps = {}
        for prompt in args.prompts:
            mlp = NeuralStyleField(args, input_dim=self.input_dim).to(device)
            mlp.reset_weights()

            self.mlps[prompt] = mlp

    def forward(self, vertices):
        # Prop. through MLPs

        for prompt, mlp in self.mlps.items():
            pred_rgb_per_prompt, pred_normal_per_prompt = mlp(vertices)
            pred_rgb_masked = pred_rgb_per_prompt*(1- self.masks[prompt])
            pred_normal_masked = pred_normal_per_prompt*(1- self.masks[prompt])

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
        