
import torch
import torch.nn as nn
import numpy as np
from utils.utils import device

class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
            2)  # no need to reduce d or to check cases
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)  ## this layer means pure ffn without progress.
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha

# ================== POSITIONAL ENCODERS =============================
class FourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude
        B = torch.randn((num_input_channels, mapping_size)) * scale
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))
        self._B = torch.stack(B_sort)  # for sape

    def forward(self, x):
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        res = 2 * np.pi * res
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)