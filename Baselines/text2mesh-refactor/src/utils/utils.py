import torch
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def get_homogenous_coordinates(V):
    N, D = V.shape
    bottom = torch.ones(N, device=device).unsqueeze(1)
    return torch.cat([V, bottom], dim=1)

def apply_affine(verts, A):
    verts = verts.to(device)
    verts = get_homogenous_coordinates(verts)
    A = torch.cat([A, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0)], dim=0)
    transformed_verts = A @ verts.T
    transformed_verts = transformed_verts[:-1]
    return transformed_verts.T

# Get rotation matrix about vector through origin
def getRotMat(axis, theta):
    """
    axis: np.array, normalized vector
    theta: radians
    """
    import math

    axis = axis / np.linalg.norm(axis)
    cprod = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    rot = math.cos(theta) * np.identity(3) + math.sin(theta) * cprod + \
          (1 - math.cos(theta)) * np.outer(axis, axis)
    return rot
