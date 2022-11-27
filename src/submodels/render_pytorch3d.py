import torch
import numpy as np

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings,
    MeshRenderer, MeshRasterizer, HardPhongShader, TexturesVertex
)
from pytorch3d.structures import Meshes

from src.models.fix_numerics import NumericsBackward
from src.utils.utils import device

class RendererPytorch3D():

    def __init__(self,
                 lights=None,
                 camera=None,
                 dim=(224, 224)):

        self.raster_settings = RasterizationSettings(
        image_size=dim,
        blur_radius=0.0,
        faces_per_pixel=1)

        self.num_backward = NumericsBackward.apply

    def render_front_views(self, mesh, num_views=8, std=8, center_elev=0, center_azim=0, show=False, lighting=True,
                           background=None, mask=False, return_views=False):
        """
        Returns images, shape [B, 3, H, W] or [B, 4, H, W]
        """
        DIST = 2.

        verts = mesh.vertices
        faces = mesh.faces
        colors = mesh.vertex_colors

        textures = TexturesVertex([colors for i in range(num_views)])
        pytorch3d_mesh = Meshes([verts for i in range(num_views)], [faces for i in range(num_views)], textures)

        elev = torch.cat((torch.tensor([center_elev]), torch.randn(num_views - 1) * np.pi / std + center_elev))
        azim = torch.cat((torch.tensor([center_azim]), torch.randn(num_views - 1) * 2 * np.pi / std + center_azim))

        R, T = look_at_view_transform(DIST, elev, azim, degrees=False)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)
        )

        images = renderer(pytorch3d_mesh)        
        images = images[:,:,:,:3].permute(0, 3, 1, 2)

        if return_views == True:
            return images, elev, azim
        else:
            return images