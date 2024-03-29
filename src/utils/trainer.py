import clip
import torch
import torchvision
import os
from src.utils.utils import device

class Trainer():
    
    def __init__(self, model, network_input, prompts, optimizer, lr_scheduler, loss_func):
        self.model = model
        self.network_input = network_input

        self.encoded_text_prompts = []
        for prompt in prompts:
            prompt_token = clip.tokenize([prompt]).to(device)
            self.encoded_text_prompts.append(self.model.clip_with_augs.clip_model.encode_text(prompt_token))

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.loss_func = loss_func
    
    def training_step(self, i, wandb, clipavg = "view", save_renders=True, save_dir="./"):
        self.optimizer.zero_grad()
        out_dict = self.model(self.network_input)
        losses_dict_per_prompt = self.loss_func(out_dict, self.encoded_text_prompts, self.model.args, clipavg)

        for losses_dict in losses_dict_per_prompt.values():
            for loss in losses_dict.values():
                if isinstance(loss, torch.Tensor):
                    loss.backward(retain_graph=True)

        self.optimizer.step()
        
        if self.model.args.optimize_gauss_estimator:
            params = self.model.gauss_estimator.parameters()
        elif self.model.args.optimize_learned_labels:
            params = [self.model.learned_labels]
        else:
            params = self.model.mlp.parameters()

        for param in params:
            param.requires_grad = True
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if save_renders and i % 100 == 0:
            img_path = os.path.join(save_dir, 'iter_{}.jpg'.format(i))
            torchvision.utils.save_image(out_dict["rendered_images"], img_path)
            wandb.log({'images': wandb.Image(img_path)}, step=i)            

        with torch.no_grad():
            return_dict = {}
            for key, loss in losses_dict.items():
                if isinstance(loss, torch.Tensor):
                    wandb.log({f"loss_{key}": loss.item()}, step=i)
                    return_dict[key] = loss.item()

            return return_dict
            