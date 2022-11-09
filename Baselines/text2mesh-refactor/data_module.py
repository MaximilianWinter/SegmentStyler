from pathlib import Path

import pytorch_lightning as pl
from torchvision import transforms
from mesh import Mesh
import torch
from Normalization import MeshNormalizer
from helpers import get_background, check_previous_run

from utils import device 
import os

def get_augumentations(args, res, clip_normalizer):
        # Augmentation settings
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(1, 1)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        
        # Augmentations for normal network
        curcrop = args.normmincrop if args.cropforward else args.normmaxcrop
            
        normaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5)
        ])
        cropiter = 0
        cropupdate = 0
        if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
            cropiter = round(args.n_iter / (args.cropsteps + 1))
            cropupdate = (args.maxcrop - args.mincrop) / cropiter

            if not args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        displaugment_transform = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(args.normmincrop, args.normmincrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer
        ])
        
        return augment_transform, normaugment_transform, displaugment_transform

def get_mesh_with_attrb(args):
    objbase, _ = os.path.splitext(os.path.basename(args.obj_path))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    check_previous_run(objbase, args)
    mesh = Mesh(args.obj_path)
    MeshNormalizer(mesh)()
    prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)
    background = get_background(args.background, torch, device)  
    return mesh, (prior_color, background, objbase)

class MeshDataModule(pl.LightningDataModule):
    pass