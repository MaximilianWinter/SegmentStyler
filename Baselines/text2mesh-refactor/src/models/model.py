
import clip

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

import numpy as np
import random

from PIL import Image
import os
from pathlib import Path

from data.mesh import Mesh
from submodels.render import Renderer
from submodels.neural_style_field import NeuralStyleField
from utils.Normalization import MeshNormalizer
from utils.utils import device


class Text2Mesh():
    def __init__(self, args):
        super().__init__()

        self.set_seed()
        self.normweight = 1.0
        self.args = args

        self.res = self.get_render_resolution()
        self.renderer = Renderer(dim=(self.res, self.res))
        self.mesh = self.get_mesh()
        self.store_mesh_params()
        self.clip_model, self.preprocess, self.clip_transform = self.get_clip()
        self.augment_transform, self.normaugment_transform, \
            self.displaugment_transform = self.get_augumentations()
        self.mlp = self.get_neural_style_network()

        self.encoded_text, self.norm_encoded = self.get_text_embeddings()
        self.encoded_image = self.get_img_embeddings()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = nn.functional.mse_loss(x_hat, x)
        # # Logging to TensorBoard by default
        # self.log("train_loss", loss)
        # return loss
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.mlp.parameters(
        ), self.args.learning_rate, weight_decay=self.args.decay)
        activate_scheduler = self.args.lr_decay < 1 and self.args.decay_step > 0 and not self.args.lr_plateau

        lr_scheduler = None
        if activate_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optim, step_size=self.args.decay_step, gamma=self.args.lr_decay)

        return [optim], [lr_scheduler]

    def forward(self, x):
        pass

    def validation_setp(self, batch, batch_idx):
        pass

    def prepare_data(self):
        pass

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def set_seed(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def get_render_resolution(self):
        """
        Sets output resolution depending on model type
        """
        self.res = 224
        if self.clipmodel == "ViT-L/14@336px":
            self.res = 336
        if self.clipmodel == "RN50x4":
            self.res = 288
        if self.clipmodel == "RN50x16":
            self.res = 384
        if self.clipmodel == "RN50x64":
            self.res = 448

        return self.res

    def get_background(background, torch=None):
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

    def check_previous_run(self):
        if (not self.args.overwrite) and os.path.exists(os.path.join(self.args.output_dir, "loss.png")) and \
                os.path.exists(os.path.join(self.args.output_dir, f"{self.objbase}_final.obj")):
            print(f"Already done with {self.args.output_dir}")
            exit()
        elif self.args.overwrite and os.path.exists(os.path.join(self.args.output_dir, "loss.png")) and \
                os.path.exists(os.path.join(self.args.output_dir, f"{self.objbase}_final.obj")):
            import shutil
            for filename in os.listdir(self.args.output_dir):
                file_path = os.path.join(self.args.output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def get_clip(self):
        self.n_augs = self.args.n_augs
        self.dir = self.args.output_dir
        self.clip_model, self.preprocess = clip.load(
            self.args.clipmodel, device, jit=self.args.jit)

        self.clip_normalizer = self.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        self.clip_transform = self.transforms.Compose([
            self.transforms.Resize((self.res, self.res)),
            self.clip_normalizer
        ])
        return self.clip_model, self.preprocess, self.clip_transform

    def get_neural_style_network(self):
        # MLP Settings
        self.input_dim = 6 if self.args.input_normals else 3
        if self.args.only_z:
            self.input_dim = 1
        self.mlp = NeuralStyleField(self.args.sigma, self.args.depth, self.args.width, 'gaussian', self.args.colordepth, self.args.normdepth,
                                    self.args.normratio, self.args.clamp, self.args.normclamp, niter=self.args.n_iter,
                                    progressive_encoding=self.args.pe, input_dim=self.input_dim, exclude=self.args.exclude).to(device)
        self.mlp.reset_weights()

        return self.mlp

    def get_text_embeddings(self):
        # Constrain all sources of randomness
        if not self.args.no_prompt:
            if self.args.prompt:
                prompt = ' '.join(self.args.prompt)
                prompt_token = clip.tokenize([prompt]).to(device)
                self.encoded_text = self.clip_model.encode_text(prompt_token)

                # Save prompt
                with open(os.path.join(dir, prompt), "w") as f:
                    f.write("")

                # Same with normprompt
                self.norm_encoded = self.encoded_text

        if self.args.normprompt is not None:
            prompt = ' '.join(self.args.normprompt)
            prompt_token = clip.tokenize([prompt]).to(device)
            self.norm_encoded = self.clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
                f.write("")

            return self.encoded_text, self.norm_encoded

    def get_img_embeddings(self):

        if self.args.image:
            img = Image.open(self.args.image)
            img = self.preprocess(img).to(device)
            self.encoded_image = self.clip_model.encode_image(img.unsqueeze(0))
            if self.args.no_prompt:
                self.norm_encoded = self.encoded_image

        return self.encoded_image

    def get_augumentations(self):
        # Augmentation settings
        self.augment_transform = self.transforms.Compose([
            self.transforms.RandomResizedCrop(self.res, scale=(1, 1)),
            self.transforms.RandomPerspective(
                fill=1, p=0.8, distortion_scale=0.5),
            self.clip_normalizer
        ])

        # Augmentations for normal network
        curcrop = self.args.normmincrop if self.args.cropforward else self.args.normmaxcrop

        self.normaugment_transform = self.transforms.Compose([
            self.transforms.RandomResizedCrop(
                self.res, scale=(curcrop, curcrop)),
            self.transforms.RandomPerspective(
                fill=1, p=0.8, distortion_scale=0.5)
        ])
        cropiter = 0
        cropupdate = 0
        if self.args.normmincrop < self.args.normmaxcrop and self.args.cropsteps > 0:
            cropiter = round(self.args.n_iter / (self.args.cropsteps + 1))
            cropupdate = (self.args.maxcrop - self.args.mincrop) / cropiter

            if not self.args.cropforward:
                cropupdate *= -1

        # Displacement-only augmentations
        self.displaugment_transform = self.transforms.Compose([
            self.transforms.RandomResizedCrop(self.res, scale=(
                self.args.normmincrop, self.args.normmincrop)),
            self.transforms.RandomPerspective(
                fill=1, p=0.8, distortion_scale=0.5),
            self.clip_normalizer
        ])

        return self.augment_transform, self.normaugment_transform, self.displaugment_transform

    def make_output_dir(self):
        self.objbase, _ = os.path.splitext(
            os.path.basename(self.args.obj_path))
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

    def load_mesh(self):
        self.mesh = Mesh(self.args.obj_path)
        MeshNormalizer(self.mesh)()
        return self.mesh

    def store_mesh_params(self):
        self.prior_color = torch.full(
            size=(self.mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)
        self.background = self.get_background(
            self.args.background, torch, device)

    def get_mesh(self):
        self.make_output_dir()
        self.check_previous_run()
        self.mesh = self.load_mesh()
        return self.mesh
