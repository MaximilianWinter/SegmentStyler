from pathlib import Path
import torch
import numpy as np
from kaolin.metrics.pointcloud import sided_distance
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.data.mesh import Mesh
from src.partglot.wrapper import PartSegmenter
from src.utils.utils import gaussian3D, device
from src.partglot.utils.partglot_bspnet_preprocess import normalize_pointcloud
from src.helper.preprocessing import remesh_per_part


class PartGlotData(torch.utils.data.Dataset):
    dataset_path = Path("/mnt/hdd/PartGlotData")
    shapenet_path = Path("/mnt/hdd/ShapeNetCore.v2")
    label_mapping = {"back": 0, "seat": 1, "leg": 2, "arm": 3}
    rev_label_mapping = {0: "back", 1: "seat", 2: "leg", 3: "arm"}
    label_color = {0: "r", 1: "g", 2: "b", 3: "m"}

    def __init__(self, prompts, *args, **kwargs):
        """
        Constructor.
        @param prompts: list of strings, the prompts
        """
        super().__init__()
        self.items = (
            PartGlotData.dataset_path.joinpath("data_mapping_chair_bsp.txt")
            .read_text()
            .splitlines()
        )

        self.prompts = prompts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pg_id, synset_id, item_id = self.items[index].split("/")
        mesh = PartGlotData.get_mesh(synset_id, item_id)
        masks, labels = PartGlotData.get_masks(
            mesh, synset_id, item_id, int(pg_id), self.prompts
        )
        weights, sigmas, coms = PartGlotData.get_gaussian_weights(mesh, masks)
        return {
            "name": f"{synset_id}-{item_id}",
            "mesh": mesh,
            "masks": masks,
            "weights": weights,
            "sigmas": sigmas,
            "coms": coms,
            "labels": labels,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        batch["masks"] = {
            key: batch["masks"][key].to(device) for key in batch["masks"].keys()
        }
        batch["weights"] = {
            key: batch["weights"][key].to(device) for key in batch["weights"].keys()
        }
        batch["sigmas"] = {
            key: batch["sigmas"][key].to(device) for key in batch["sigmas"].keys()
        }
        batch["coms"] = {
            key: batch["coms"][key].to(device) for key in batch["coms"].keys()
        }

    @staticmethod
    def get_mesh(synset_id, item_id):
        mesh_path = PartGlotData.dataset_path.joinpath(
            f"{synset_id}/{item_id}/mesh.obj"
        )
        if not mesh_path.exists():
            print("Do remeshing...")
            shapenet_mesh_path = PartGlotData.shapenet_path.joinpath(
                f"{synset_id}/{item_id}/models/model_normalized.obj"
            )
            remesh_per_part(shapenet_mesh_path, mesh_path, remesh_iterations=5)
            print("Finished remeshing.")

        mesh = Mesh(
            str(PartGlotData.dataset_path.joinpath(f"{synset_id}/{item_id}/mesh.obj")),
            use_trimesh=False,
        )
        return mesh

    @staticmethod
    def get_masks(mesh, synset_id, item_id, pg_id, prompts):
        part_names = []
        for part in PartGlotData.label_mapping.keys():
            for prompt in prompts:
                if part in prompt:
                    part_names.append(part)
                    break
        ps = PartSegmenter(
            part_names=PartGlotData.label_mapping.keys(),
            partglot_data_dir="/mnt/hdd/PartGlotData/",
            partglot_model_path="models/pn_agnostic.ckpt",
        )
        _, _, partmaps = ps.run_from_ref_data(sample_idx=pg_id, use_sseg_gt=True)
        pg_pc = None
        pg_labels = None

        for pn, pc in zip(part_names, partmaps):
            print(pn, pc.shape, np.unique(pc, axis=0).shape)
            unique_array = np.unique(pc, axis=0)
            if pg_pc is None:
                pg_pc = unique_array
                pg_labels = (
                    np.ones_like(unique_array, dtype=int)
                    * PartGlotData.label_mapping[pn]
                )
            else:
                pg_pc = np.vstack([pg_pc, unique_array])
                pg_labels = np.vstack(
                    [
                        pg_labels,
                        np.ones_like(unique_array, dtype=int)
                        * PartGlotData.label_mapping[pn],
                    ]
                )
        normalized_pg_pc = normalize_pointcloud(pg_pc)["pc"]
        normalized_vertices = normalize_pointcloud(
            mesh.vertices.double().cpu().numpy()
        )["pc"]
        p2 = torch.tensor(normalized_pg_pc).unsqueeze(0).to(device)
        p1 = torch.tensor(normalized_vertices).unsqueeze(0).to(device)
        _, indices = sided_distance(p1, p2)
        labels = pg_labels[indices.cpu()][0, :, 0]

        masks = {}
        for prompt in prompts:
            if ("legs" in prompt) and ("legs" not in PartGlotData.label_mapping.keys()):
                parts = ["leg"]
            else:
                parts = [
                    part for part in PartGlotData.label_mapping.keys() if part in prompt
                ]
            mask = torch.ones(len(mesh.vertices), 3)
            for part in parts:
                mask[labels == PartGlotData.label_mapping[part]] = 0
            masks[prompt] = mask

        return masks, labels

    @staticmethod
    def get_gaussian_weights(mesh, masks):
        """
        @param masks: dict with prompts as keys and masks as keys
        @returns: tuple of normalized_weights, dict of normalized gaussian weights, sigmas and coms (both dicts)
        """
        normalized_weights = {}
        sum_of_weights = None
        sigmas = {}
        coms = {}

        for prompt, mask in masks.items():
            inv_mask = 1 - mask
            part_vertices = mesh.vertices[inv_mask[:, 0].bool()].detach()

            COM = torch.mean(part_vertices, dim=0)
            Sigma = (
                (part_vertices - COM).T
                @ (part_vertices - COM)
                / (part_vertices.shape[0] - 1)
            )
            gauss_weight = gaussian3D(mesh.vertices, COM, Sigma)

            weight = torch.zeros_like(inv_mask)
            for i in range(weight.shape[1]):
                weight[:, i] = gauss_weight
            normalized_weights[prompt] = weight

            if sum_of_weights is None:
                sum_of_weights = weight.clone()
            else:
                sum_of_weights += weight

            sigmas[prompt] = Sigma
            coms[prompt] = COM

        for prompt in normalized_weights.keys():
            normalized_weights[prompt][sum_of_weights != 0] /= sum_of_weights[
                sum_of_weights != 0
            ]

        return normalized_weights, sigmas, coms

    @staticmethod
    def visualize_predicted_maps(points, labels, path):
        label_colors = [PartGlotData.label_color[label] for label in labels]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        sp = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=label_colors)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                ls="",
                color=PartGlotData.label_color[label],
                label=PartGlotData.rev_label_mapping[label],
            )
            for label in np.unique(labels)
        ]
        ax.legend(handles=legend_elements, loc="right")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=30, azim=-120, vertical_axis="y")
        ax.set_title("PartGlot predictions")
        plt.savefig(path)
