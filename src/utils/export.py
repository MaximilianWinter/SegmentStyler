import kaolin as kal
import torch
import numpy as np
from torchvision import transforms
import os
from pathlib import Path

from src.submodels.render import Renderer
from src.utils.Normalization import MeshNormalizer
from src.utils.utils import device


def export_final_results(args, dir, losses, mesh, mlp, network_input, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        if args.optimize_displacement:
            mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal
        else:
            mesh.vertices = vertices.detach().cpu()

        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)

        # Run renders
        if args.save_render:
            save_rendered_results(args, dir, final_color, mesh)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))


def save_rendered_results(args, dir, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))
    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, args.frontview_center[1], args.frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))