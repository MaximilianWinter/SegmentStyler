import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin as kal
import copy
import jstyleson

from src.models.original_model import Text2MeshOriginal 
from src.submodels.clip_with_augs import CLIPWithAugs
from src.submodels.render import Renderer
from src.submodels.neural_style_field import NeuralStyleField
from src.utils.render import get_render_resolution
from src.utils.Normalization import MeshNormalizer
from src.utils.utils import device


class Text2MeshPenalizingLoss(Text2MeshOriginal):
    
    def get_color_reg(self, pred_rgb):
        """
        Extracts ground truth color regularizer
        """        
        with open(self.args.mask_path) as fp:
            mesh_metadata = jstyleson.load(fp)
            
        mask = torch.ones_like(pred_rgb)
        
        for start, finish in mesh_metadata["mask_vertices"].values():
            mask[start:finish] = 0 
        
        color_reg = torch.sum(pred_rgb**2*mask) # penalizing term, to be added to the loss
        
        return color_reg
        