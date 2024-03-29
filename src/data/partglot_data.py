from pathlib import Path
import torch
import numpy as np
from kaolin.metrics.pointcloud import sided_distance
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from kmeans_pytorch import kmeans
import copy

from src.data.mesh import Mesh
from src.partglot.wrapper import PartSegmenter
from src.utils.utils import gaussian3D, device
from src.utils.Normalization import MeshNormalizer
from src.partglot.utils.partglot_bspnet_preprocess import (
    normalize_pointcloud,
    rotate_pointcloud,
)
from src.helper.preprocessing import remesh_per_part
from src.helper.paths import GLOBAL_DATA_PATH


class PartGlotData(torch.utils.data.Dataset):
    dataset_path = GLOBAL_DATA_PATH.joinpath("PartGlotData")
    shapenet_path = GLOBAL_DATA_PATH.joinpath("ShapeNetCore.v2")
    partseg_gt_path = GLOBAL_DATA_PATH.joinpath("shapenetcore_partanno_segmentation_benchmark_v0")

    label_mapping = {"back": 0, "seat": 1, "leg": 2, "arm": 3}
    rev_label_mapping = {0: "back", 1: "seat", 2: "leg", 3: "arm"}
    label_color = {0: "r", 1: "g", 2: "b", 3: "m"}

    def __init__(
        self,
        prompts,
        *args,
        return_keys=[
            "mesh",
            "masks",
            "weights",
            "sigmas",
            "coms",
            "labels",
            "gt_labels",
            "gt_masks",
        ],
        **kwargs,
    ):
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
        self.return_keys = return_keys

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pg_id, synset_id, item_id = self.items[index].split("/")
        mesh = masks = weights = sigmas = coms = labels = gt_labels = gt_masks = None
        if "mesh" in self.return_keys:
            mesh = PartGlotData.get_mesh(synset_id, item_id)
            if "gt_labels" in self.return_keys:
                gt_labels = PartGlotData.get_gt_labels(mesh, synset_id, item_id)
                if "gt_masks" in self.return_keys:
                    gt_masks = PartGlotData.masks_from_labels(
                        mesh, gt_labels, self.prompts
                    )
            if "masks" in self.return_keys or "labels" in self.return_keys:
                masks, labels = PartGlotData.get_masks(
                    mesh, synset_id, item_id, int(pg_id), self.prompts
                )
                if (
                    "weights" in self.return_keys
                    or "sigmas" in self.return_keys
                    or "coms" in self.return_keys
                ):
                    weights, sigmas, coms = PartGlotData.get_gaussian_weights(
                        mesh, masks
                    )

        return {
            "name": f"{synset_id}-{item_id}",
            "mesh": mesh,
            "masks": masks,
            "weights": weights,
            "sigmas": sigmas,
            "coms": coms,
            "labels": labels,
            "gt_labels": gt_labels,
            "gt_masks": gt_masks,
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
            key: batch["sigmas"][key].to(device) if not isinstance(batch["sigmas"][key], list) else [sigma.to(device) for sigma in batch["sigmas"][key]] for key in batch["sigmas"].keys()
        }
        batch["coms"] = {
            key: batch["coms"][key].to(device) if not isinstance(batch["coms"][key], list) else [com.to(device) for com in batch["coms"][key]] for key in batch["coms"].keys()
        }
        batch["gt_masks"] = {
            key: batch["gt_masks"][key].to(device) for key in batch["gt_masks"].keys()
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
        print(f"Using part names: {part_names}")
        ps = PartSegmenter(
            part_names=part_names,
            partglot_data_dir=PartGlotData.dataset_path,
            partglot_model_path=PartGlotData.dataset_path.joinpath("pn_aware.ckpt"),
            prompts=prompts,  # if this is not None, partglot will use the prompts instead of the template sentence
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

        masks = PartGlotData.masks_from_labels(mesh, labels, prompts)

        return masks, labels

    @staticmethod
    def masks_from_labels(mesh, labels, prompts):
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

        return masks

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

        mesh = copy.deepcopy(mesh)
        MeshNormalizer(mesh)()

        for prompt, mask in masks.items():
            inv_mask = 1 - mask
            part_vertices = mesh.vertices[inv_mask[:, 0].bool()].detach()
            if "arm" in prompt or "leg" in prompt:
                device = part_vertices.device
                n_clusters = 2 if "arm" in prompt else 4
                cluster_ids, cluster_centers = kmeans(X=part_vertices, num_clusters=n_clusters, distance='euclidean', device=device)
                cluster_centers = cluster_centers.to(device)
                cluster_ids = cluster_ids.to(device)
                gauss_weights = []
                sigma_list = []
                for i, center in enumerate(cluster_centers):
                    sub_part_vertices = part_vertices[cluster_ids == i]
                    Sigma = (sub_part_vertices - center).T@(sub_part_vertices - center)/(sub_part_vertices.shape[0] - 1)
                    sigma_list.append(Sigma)
                    gauss_weights.append(gaussian3D(mesh.vertices, center, Sigma).unsqueeze(0))
                
                gauss_weight = torch.sum(torch.cat(gauss_weights, dim=0), dim=0)
                sigmas[prompt] = sigma_list
                coms[prompt] = list(cluster_centers)
            else:
                center = torch.mean(part_vertices, dim=0)
                Sigma = (
                    (part_vertices - center).T
                    @ (part_vertices - center)
                    / (part_vertices.shape[0] - 1)
                )
                gauss_weight = gaussian3D(mesh.vertices, center, Sigma)

                sigmas[prompt] = [Sigma]
                coms[prompt] = [center]

            weight = torch.zeros_like(inv_mask)
            for i in range(weight.shape[1]):
                weight[:, i] = gauss_weight
            normalized_weights[prompt] = weight

            if sum_of_weights is None:
                sum_of_weights = weight.clone()
            else:
                sum_of_weights += weight

        for prompt in normalized_weights.keys():
            normalized_weights[prompt][sum_of_weights != 0] /= sum_of_weights[
                sum_of_weights != 0
            ]

        return normalized_weights, sigmas, coms

    @staticmethod
    def get_gt_labels(mesh, synset_id, item_id):
        points_list = (
            PartGlotData.partseg_gt_path.joinpath(f"{synset_id}/points/{item_id}.pts")
            .read_text()
            .splitlines()
        )
        labels_list = (
            PartGlotData.partseg_gt_path.joinpath(
                f"{synset_id}/points_label/{item_id}.seg"
            )
            .read_text()
            .splitlines()
        )
        labels = (
            np.array(labels_list, dtype=int) - 1
        )  # the gt labels in the dataset are from 1 to 4, we want them to be from 0 to 3
        points = np.zeros((len(points_list), 3))
        for i, pts in enumerate(points_list):
            points[i, :] = pts.split(" ")

        normalized_pc = normalize_pointcloud(rotate_pointcloud(points))["pc"]
        normalized_vertices = normalize_pointcloud(
            mesh.vertices.double().cpu().numpy()
        )["pc"]
        p2 = torch.tensor(normalized_pc).unsqueeze(0).to(device)
        p1 = torch.tensor(normalized_vertices).unsqueeze(0).to(device)
        _, indices = sided_distance(p1, p2)
        mesh_gt_labels = labels[indices.cpu()][0, :]

        return mesh_gt_labels

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
