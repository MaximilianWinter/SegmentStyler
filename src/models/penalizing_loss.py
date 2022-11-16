import torch

from src.models.original_model import Text2MeshOriginal 

class Text2MeshPenalizingLoss(Text2MeshOriginal):
    
    def get_color_reg(self, pred_rgb):
        """
        Extracts ground truth color regularizer
        """        
        mask = self.load_mask(pred_rgb)
        
        color_reg = torch.sum(pred_rgb**2*mask) # penalizing term, to be added to the loss
        
        return color_reg
        