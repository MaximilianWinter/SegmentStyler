import tqdm
import clip
import copy
import torch

from data.mesh import Mesh
from utils.Normalization import MeshNormalizer
from models.original_model import Text2MeshOriginal
from utils.utils import device

class Trainer():
    
    def __init__(self, model, network_input, prompt):
        self.model = model
        self.network_input = network_input

        prompt_token = clip.tokenize([prompt]).to(device)
        self.encoded_text = self.model.clip_with_augs.clip_model.encode_text(prompt_token)
    
    def training_step(self):
        pass


def train(args):
    # LOAD MESH AND MODEL
    base_mesh = Mesh(args.obj_path)
    MeshNormalizer(base_mesh)()
    text2mesh_model = Text2MeshOriginal(args, base_mesh)

    # NETWORK INPUT
    vertices = copy.deepcopy(base_mesh.vertices)
    network_input = copy.deepcopy(vertices)   
    if args.symmetry == True:
        network_input[:,2] = torch.abs(network_input[:,2])

    if args.standardize == True:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)

    # TEXT PROMPT
    if args.prompt:
        prompt = ' '.join(args.prompt)
    else:
        raise ValueError("No prompt given.")

    trainer = Trainer(text2mesh_model, network_input, prompt)

    for i in tqdm(range(args.n_iter)):
        trainer.training_step()