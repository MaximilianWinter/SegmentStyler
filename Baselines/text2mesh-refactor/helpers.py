from PIL import Image
from neural_style_field import NeuralStyleField
import clip
import os
import numpy as np
import random
from torchvision import transforms
from utils import device


def get_resolution(clipmodel=None):
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


def check_previous_run(objbase, args):
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.output_dir}")
        exit()
    elif args.overwrite and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_clip(args, res):
    n_augs = args.n_augs
    dir = args.output_dir
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)

    clip_normalizer = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # CLIP Transform
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])
    return clip_model, preprocess, clip_transform


def get_neural_style_network(args):
    # MLP Settings
    input_dim = 6 if args.input_normals else 3
    if args.only_z:
        input_dim = 1
    mlp = NeuralStyleField(args.sigma, args.depth, args.width, 'gaussian', args.colordepth, args.normdepth,
                           args.normratio, args.clamp, args.normclamp, niter=args.n_iter,
                           progressive_encoding=args.pe, input_dim=input_dim, exclude=args.exclude).to(device)
    mlp.reset_weights()

    return mlp


def get_text_embeddings(args, clip_model):
    # Constrain all sources of randomness
    if not args.no_prompt:
        if args.prompt:
            prompt = ' '.join(args.prompt)
            prompt_token = clip.tokenize([prompt]).to(device)
            encoded_text = clip_model.encode_text(prompt_token)

            # Save prompt
            with open(os.path.join(dir, prompt), "w") as f:
                f.write("")

            # Same with normprompt
            norm_encoded = encoded_text

    if args.normprompt is not None:
        prompt = ' '.join(args.normprompt)
        prompt_token = clip.tokenize([prompt]).to(device)
        norm_encoded = clip_model.encode_text(prompt_token)

        # Save prompt
        with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
            f.write("")

        return encoded_text, norm_encoded


def get_img_embeddings(self):

    if self.args.image:
        img = Image.open(self.args.image)
        img = self.preprocess(img).to(device)
        encoded_image = self.clip_model.encode_image(img.unsqueeze(0))
        if self.args.no_prompt:
            self.norm_encoded = encoded_image

    return encoded_image
