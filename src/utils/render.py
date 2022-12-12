import torch
import kaolin as kal

def get_camera_from_view(elev, azim, r=3.0):
    x = r * torch.cos(azim) * torch.sin(elev)
    y = r * torch.sin(azim) * torch.sin(elev)
    z = r * torch.cos(elev)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_camera_from_view_and_center(elev, azim, r=3.0, center_point=torch.tensor([0.0, 0.0, 0.0])):
    x = r * torch.cos(elev) * torch.cos(azim) + center_point[0]
    y = r * torch.sin(elev) + center_point[1]
    z = r * torch.cos(elev) * torch.sin(azim) + center_point[2]
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = center_point.cpu()-pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj


def get_render_resolution(clipmodel=None):
        """
        Sets output resolution depending on model type
        """
        res = 224
        if clipmodel == "ViT-L/14@336px":
            res = 336
        if clipmodel == "RN50x4":
            res = 288
        if clipmodel == "RN50x16":
            res = 384
        if clipmodel == "RN50x64":
            res = 448

        return res