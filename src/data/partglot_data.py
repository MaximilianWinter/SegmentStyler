from pathlib import Path
import torch
import numpy as np
from kaolin.metrics.pointcloud import sided_distance
import trimesh

from src.data.mesh import Mesh
from src.partglot.wrapper import PartSegmenter
from src.utils.utils import gaussian3D, device
from src.partglot.utils.partglot_bspnet_preprocess import normalize_pointcloud


class PartGlotData(torch.utils.data.Dataset):
    dataset_path = Path("data/partglot_data")

    def __init__(self, prompts, noisy=False):
        """
        Constructor.
        @param prompts: list of strings, the prompts
        @param noisy: boolean, not used
        """
        super().__init__()
        self.items = (
            PartGlotData.dataset_path.joinpath("data_mapping.txt")
            .read_text()
            .splitlines()
        )

        self.prompts = prompts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        pg_id, synset_id, item_id = self.items[index].split("/")
        mesh = PartGlotData.get_mesh(synset_id, item_id)
        masks, labels, tri_mesh = PartGlotData.get_masks(
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
            "tri_mesh": tri_mesh,
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
        mesh = Mesh(
            str(PartGlotData.dataset_path.joinpath(f"{synset_id}/{item_id}/mesh.obj")),
            use_trimesh=True,
        )
        return mesh

    @staticmethod
    def get_masks(mesh, synset_id, item_id, pg_id, prompts):
        with open(
            PartGlotData.dataset_path.joinpath(f"{synset_id}/{item_id}/mesh.obj")
        ) as fp:
            mesh_dict = trimesh.exchange.obj.load_obj(
                fp, include_color=False, include_texture=False
            )
            tri_mesh = trimesh.Trimesh(**mesh_dict)
        part_names = ["back", "seat", "leg", "arm"]
        ps = PartSegmenter(
            part_names=part_names,
            partglot_data_dir="/mnt/hdd/PartGlotData/",
            partglot_model_path="models/pn_agnostic.ckpt",
        )
        _, _, partmaps = ps.run_from_ref_data(sample_idx=pg_id, use_sseg_gt=True)
        pg_pc = None
        pg_labels = None
        label_mapping = {"back": 0, "seat": 1, "leg": 2, "arm": 3}
        for pn, pc in zip(part_names, partmaps):
            print(pn, pc.shape, np.unique(pc, axis=0).shape)
            unique_array = np.unique(pc, axis=0)
            if pg_pc is None:
                pg_pc = unique_array
                pg_labels = np.ones_like(unique_array, dtype=int) * label_mapping[pn]
            else:
                pg_pc = np.vstack([pg_pc, unique_array])
                pg_labels = np.vstack(
                    [
                        pg_labels,
                        np.ones_like(unique_array, dtype=int) * label_mapping[pn],
                    ]
                )
        normalized_pg_pc = normalize_pointcloud(pg_pc)["pc"]
        normalized_vertices = normalize_pointcloud(tri_mesh.vertices)["pc"]
        p2 = torch.tensor(normalized_pg_pc).unsqueeze(0).to(device)
        p1 = torch.tensor(normalized_vertices).unsqueeze(0).to(device)
        _, indices = sided_distance(p1, p2)
        labels = pg_labels[indices.cpu()][0, :, 0]

        masks = {}
        for prompt in prompts:
            if ("legs" in prompt) and ("legs" not in label_mapping.keys()):
                parts = ["leg"]
            else:
                parts = [part for part in label_mapping.keys() if part in prompt]
            mask = torch.ones(len(mesh.vertices), 3)
            for part in parts:
                mask[labels == label_mapping[part]] = 0
            masks[prompt] = mask

        return masks, labels, tri_mesh

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
