import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils.encoding import FourierFeatureTransform, ProgressiveEncoding


class GaussEstimator(nn.Module):

    def __init__(self, args, input_dim=3, output_dim=3, n_prompts=3) -> None:
        """
        @param args:
        @param input_dim: int, specifying input dim
        @param output_dim: int, specifying output dimension = number of Gaussians for which parameters should be estimated
        """
        super().__init__()

        self.layers = nn.Sequential(
            FourierFeatureTransform(input_dim, args.width, args.sigma, args.exclude),
            ProgressiveEncoding(mapping_size=args.width, T=args.n_iter, d=input_dim) if args.pe else nn.Identity(),
            nn.Linear(args.width * 2 + input_dim, args.width),
            nn.ReLU(),
            nn.Linear(args.width, args.width),
            nn.ReLU(),
            nn.Linear(args.width, args.width),
            nn.ReLU(),
            nn.Linear(args.width, output_dim)
        )

        self.one_input = torch.ones((n_prompts,output_dim))

        print("### GaussEstimator ### \n", self.layers)

    def reset_weights(self):
        self.layers[-1].weight.data.zero_()
        self.layers[-1].bias.data.zero_()

    def forward(self, x):
        x = self.layers(x)
        mu_displacements = F.tanh(x)/2. # between -0.5 and +0.5
        return mu_displacements

