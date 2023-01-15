from pathlib import Path
import torch
import jstyleson

from src.data.mesh import Mesh
from src.utils.utils import gaussian3D


class PreprocessedShapeNet(torch.utils.data.Dataset):
    dataset_path = Path("data/preprocessed_shapenet")

    def __init__(self, prompts, noisy=False):
        """
        Constructor.
        @param prompts: list of strings, the prompts
        @param noisy: boolean, whether to use noisy masks or not
        """
        super().__init__()
        self.items = (
            PreprocessedShapeNet.dataset_path.joinpath("data_list.txt")
            .read_text()
            .splitlines()
        )

        self.prompts = prompts
        self.noisy = noisy

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        synset_id, item_id = self.items[index].split("/")
        mesh = PreprocessedShapeNet.get_mesh(synset_id, item_id)
        masks = PreprocessedShapeNet.get_masks(
            mesh, synset_id, item_id, self.prompts, noisy=self.noisy
        )
        weights, sigmas, coms = PreprocessedShapeNet.get_gaussian_weights(mesh, masks)
        return {
            "name": f"{synset_id}-{item_id}",
            "mesh": mesh,
            "masks": masks,
            "weights": weights,
            "sigmas": sigmas,
            "coms": coms,
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        raise NotImplementedError

    @staticmethod
    def get_mesh(synset_id, item_id):
        mesh = Mesh(
            str(
                PreprocessedShapeNet.dataset_path.joinpath(
                    f"{synset_id}/{item_id}/mesh.obj"
                )
            )
        )
        return mesh

    @staticmethod
    def get_masks(mesh, synset_id, item_id, prompts, noisy=False):
        with open(
            PreprocessedShapeNet.dataset_path.joinpath(
                f"{synset_id}/{item_id}/mask.jsonc"
            )
        ) as fp:
            mesh_metadata = jstyleson.load(fp)

        masks = {}
        for prompt in prompts:
            if ("legs" in prompt) and ("legs" not in mesh_metadata["mask_vertices"].keys()):
                parts = ["leg_1", "leg_2", "leg_3", "leg_4"]
            else:
                parts = [
                    part
                    for part in mesh_metadata["mask_vertices"].keys()
                    if part in prompt
                ]
            mask = torch.ones(len(mesh.vertices), 3)
            for part in parts:
                start, finish = mesh_metadata["mask_vertices"][part]
                mask[start:finish] = 0
                if noisy:
                    n_tot = mask.shape[0]
                    n = finish - start
                    random_zeros = torch.randint(0, n_tot, (n // 5,))
                    random_ones = torch.randint(0, n_tot, (n // 5,))
                    mask[random_zeros] = 0
                    mask[random_ones] = 1
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
