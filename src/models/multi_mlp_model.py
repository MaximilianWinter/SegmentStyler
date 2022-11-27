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
        raise NotImplementedError
        