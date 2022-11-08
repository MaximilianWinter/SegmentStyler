import numpy as np
import random

def get_res(clipmodel=None):
    """
    Returns output resolution depending on model type 
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

def get_background(background, torch=None, device=None):
    bg = None
    if background is not None:
        assert len(background) == 3
        bg = torch.tensor(background).to(device)
    return bg


def init_torch(torch, seed):
    # Constrain all sources of randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch