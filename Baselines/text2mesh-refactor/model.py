import os
from render import Renderer
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from helpers import get_resolution, get_neural_style_network, get_text_embeddings, get_img_embeddings
from utils import device
from data_module import get_augumentations, get_mesh_with_attrb, get_clip
import torch

# define the LightningModule


class Text2Mesh(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        import config
        self.normweight = 1.0
        self.args = args

        self.res = get_resolution(self.args.clipmodel)
        self.renderer = Renderer(dim=(self.res, self.res))
        self.mesh, attr = get_mesh_with_attrb(self.args)
        self.prior_color, self.background, self.objpath = attr
        self.clip_model, self.preprocess, self.clip_transform = get_clip(
            self.args, self.res)
        self.augment_transform, self.normaugment_transform, \
            self.displaugment_transform = get_augumentations(
                self.args, self.res, self.clip_normalizer)
        self.mlp = get_neural_style_network(self.args)

        encoded_text, norm_encoded = get_text_embeddings(self.clip_model)
        encoded_image = get_img_embeddings()

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
