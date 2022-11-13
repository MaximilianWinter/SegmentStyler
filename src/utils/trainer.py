import clip
import torch
import torchvision
import os
from utils.utils import device

class Trainer():
    
    def __init__(self, model, network_input, prompt, optimizer, lr_scheduler, loss_func):
        self.model = model
        self.network_input = network_input

        prompt_token = clip.tokenize([prompt]).to(device)
        self.encoded_text = self.model.clip_with_augs.clip_model.encode_text(prompt_token)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_func = loss_func
    
    def training_step(self, i, clipavg = "view", save_renders=True, save_dir="./"):
        self.optimizer.zero_grad()
        encoded_renders_dict, rendered_images = self.model(self.network_input)

        losses_dict = self.loss_func(encoded_renders_dict, self.encoded_text, clipavg)
        for loss in losses_dict.values():
            if loss != 0.0:
                loss.backward(retain_graph=True)

        self.optimizer.step()

        for param in self.model.mlp.parameters():
            param.requires_grad = True
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if save_renders and i % 100 == 0:
            torchvision.utils.save_image(rendered_images, os.path.join(save_dir, 'iter_{}.jpg'.format(i)))

        with torch.no_grad():
            for loss in losses_dict.values():
                if loss != 0.0:
                    return loss.item()
            
            return None