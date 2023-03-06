import torch
import trimesh
import networkx as nx
from src.data.partglot_data import PartGlotData
import torch.nn.functional as F
import numpy as np

from src.models.multi_mlp_model import Text2MeshMultiMLP
from src.utils.utils import device

class Text2MeshLabelPropagation(Text2MeshMultiMLP):
    def __init__(self, args, data_dict):
        """
        Class for performing label propagation on the mesh for PartGlot's label prediction.
        To use this module for post-processing, pre-trained weights must be loaded.
        When running main.py, the flag "optimize_learned_labels" must be set for optimizing learned_labels.
        @param args: Namespace, defining configuration
        @param data_dict: dictionary, containing all relevant data, see corresponding dataset classes for details
        """
        super().__init__(args, data_dict)
        
        # get labels from PartGlot in one-hot encoding
        # use them as initialization for the learned_labels tensor (will be optimized via GD)
        # we also relax the discrete constraint to enable optimization via GD
        self.initial_labels = F.one_hot(torch.Tensor(data_dict["labels"]).long()).to(device) # shape N, K
        self.learned_labels = self.initial_labels.clone().float() 
        self.learned_labels.requires_grad = True
        
        # get graph laplacian
        tri_mesh = trimesh.Trimesh(vertices=self.base_mesh.vertices.cpu().numpy(), faces=self.base_mesh.faces.cpu().numpy())        
        # Constructing a sparse tensor
        L_coo = nx.laplacian_matrix(tri_mesh.vertex_adjacency_graph).tocoo()
        values = L_coo.data
        indices = np.vstack((L_coo.row, L_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = L_coo.shape
        self.graph_laplacian  = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)

    def forward(self, vertices):
        """
        Forward pass. This also involves forward passes through the neural styler
        MLPs. Even though they are not used in the current implementation, we decided to keep
        them, as for future work we might implement a joint optimization based on
        label prop.'s energy and the semantic CLIP loss.
        @param vertices: torch.tensor, shape (N, 3)
        @returns: dict, containing renderings and their encodings (in CLIP space, unused),
                        a color regularization term (unused),
                        the label propagation's energy label_prop_energy
        """
        # we first normalize the labels using softmax and a temperature of 0.1
        # these normalized labels are used for calculating the energy (see below)
        normalized_labels = torch.softmax(self.learned_labels/1e-1, dim=1)

        # the weights we use for visualization/blending are essentially discretized
        # by passing the learned labels through a softmax with a temperature of 1e-9
        temperature = 1e-9
        visualized_labels = torch.softmax(self.learned_labels/temperature, dim=1)
        normalized_weights = self.get_weights_per_prompt(visualized_labels) # N, 3 each mask, for each prompt

        self.gaussian_weights = normalized_weights
        
        # Prop. through MLPs
        pred_rgb, pred_normal = self.prop_through_mlps(vertices)

        # Rendering, Augmentations and CLIP encoding per prompt
        (
            encoded_renders_dict_per_prompt,
            rendered_images_per_prompt,
            color_reg,
        ) = self.render_augment_encode(
            vertices, pred_rgb, pred_normal
        )

        lambda_flipped_labels = 1e-2
        label_prop_energy = (normalized_labels.T@torch.sparse.mm(self.graph_laplacian, normalized_labels)).diag().sum()
        if label_prop_energy < lambda_flipped_labels * (self.initial_labels != visualized_labels.int()).sum().item()/2:
            self.stop_loop = True

        return {
            "encoded_renders": encoded_renders_dict_per_prompt,
            "rendered_images": rendered_images_per_prompt,
            "color_reg": color_reg,
            "label_prop_energy": label_prop_energy
        }

    def get_weights_per_prompt(self, labels):
        """
        Helper method for converting the one-hot encoded label tensors to
        weight masks for each prompt.
        @param labels: torch.tensor, shape (N,K) with K=4 (usually)
        @returns weights: dict, containing torch.tensors of shape (N,3) as values
        """
        weights = {}
        for prompt in self.args.prompts:
            if ("legs" in prompt) and ("legs" not in PartGlotData.label_mapping.keys()):
                parts = ["leg"]
            else:
                parts = [
                    part for part in PartGlotData.label_mapping.keys() if part in prompt
                ]
            
            if len(parts) != 1:
                raise ValueError("Only works with exactly one part per prompt.")
            else:
                idx = PartGlotData.label_mapping[parts[0]]
                weights[prompt] = labels[:, idx].unsqueeze(1).repeat(1, 3) # N, 3
            
        return weights
    