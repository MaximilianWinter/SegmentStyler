import kaolin as kal
import torch
import numpy as np
from torchvision import transforms
import os
from pathlib import Path
import trimesh

from src.submodels.render import Renderer
from src.utils.Normalization import MeshNormalizer
from src.utils.utils import device, gaussian3D


def export_final_results(dir, losses, model, wandb):
    with torch.no_grad():
        if isinstance(model.mlp, torch.nn.ModuleDict):
            pred_rgb = None
            pred_normal = None
            for prompt, mlp_per_prompt in model.mlp.items():
                pred_rgb_per_prompt, pred_normal_per_prompt = mlp_per_prompt(
                    model.base_mesh_vertices
                )
                inv_mask = 1 - model.masks[prompt]
                if model.args.final_gaussian_blending:
                    weight = model.gaussian_weights[prompt]
                else:
                    weight = inv_mask
                pred_rgb_masked = pred_rgb_per_prompt * weight
                pred_normal_masked = pred_normal_per_prompt * weight

                if pred_rgb is not None:
                    pred_rgb += pred_rgb_masked
                else:
                    pred_rgb = pred_rgb_masked

                if pred_normal is not None:
                    pred_normal += pred_normal_masked
                else:
                    pred_normal = pred_normal_masked
        else:
            pred_rgb, pred_normal = model.mlp(model.base_mesh_vertices)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(
            size=(model.base_mesh.vertices.shape[0], 3), fill_value=0.5
        )
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        if model.args.optimize_displacement:
            model.base_mesh.vertices = (
                model.base_mesh_vertices.detach().cpu()
                + model.base_mesh.vertex_normals.detach().cpu() * pred_normal
            )
        else:
            model.base_mesh.vertices = model.base_mesh_vertices.detach().cpu()

        objbase, extension = os.path.splitext(os.path.basename(model.args.obj_path))
        model.base_mesh.export(
            os.path.join(dir, f"{objbase}_final.obj"), color=final_color
        )
        # Run renders
        if model.args.save_render:
            save_rendered_results(model.args, dir, final_color, model.base_mesh)

        if not model.args.no_mesh_log:
            log_mesh_to_wandb(dir, objbase, wandb)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))

    # Save final model
    torch.save(model.mlp, os.path.join(dir, "final_mlp.pt"))


def log_mesh_to_wandb(dir, objbase, wandb):
    with open(os.path.join(dir, f"{objbase}_final.obj")) as fp:
        mesh_dict = trimesh.exchange.obj.load_obj(fp)

    trimesh_mesh = trimesh.Trimesh(**mesh_dict)
    glb_output = trimesh.exchange.gltf.export_glb(trimesh_mesh)

    with open(os.path.join(dir, f"{objbase}_final.glb"), "wb") as fp:
        fp.write(glb_output)

    wandb.log(
        {"output_mesh": [wandb.Object3D(os.path.join(dir, f"{objbase}_final.glb"))]}
    )

    os.unlink(os.path.join(dir, f"{objbase}_final.glb"))


def save_rendered_results(args, dir, final_color, mesh):
    default_color = torch.full(
        size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device
    )
    mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
        default_color.unsqueeze(0), mesh.faces.to(device)
    )
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(
            np.pi / 4, 1280 / 720
        ).to(device),
        dim=(1280, 720),
    )
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(
        mesh,
        args.frontview_center[1],
        args.frontview_center[0],
        radius=2.5,
        background=torch.tensor([1, 1, 1]).to(device).float(),
        return_mask=True,
    )
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
    mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
        final_color.unsqueeze(0).to(device), mesh.faces.to(device)
    )
    img, mask = kal_render.render_single_view(
        mesh,
        args.frontview_center[1],
        args.frontview_center[0],
        radius=2.5,
        background=torch.tensor([1, 1, 1]).to(device).float(),
        return_mask=True,
    )
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))
