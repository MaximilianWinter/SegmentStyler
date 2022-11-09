import tqdm
import copy
import torch

from data.mesh import Mesh
from utils.Normalization import MeshNormalizer
from models.original_model import Text2MeshOriginal
from utils.trainer import Trainer
from utils.export import export_final_results
from utils.loss import default_loss
from utils.utils import device, report_process

def train(args):
    # LOAD MESH AND MODEL
    base_mesh = Mesh(args.obj_path)
    MeshNormalizer(base_mesh)()
    text2mesh_model = Text2MeshOriginal(args, base_mesh)

    # DEFINE OPTIMIZER
    optimizer = torch.optim.Adam(text2mesh_model.mlp.parameters(), args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)
    else:
        lr_scheduler = None
    
    # DEFINE LOSS
    loss_func = default_loss

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

    trainer = Trainer(text2mesh_model, network_input, prompt, optimizer, lr_scheduler, loss_func)

    losses = []
    loss_check = None
    for i in tqdm(range(args.n_iter)):
        loss = trainer.training_step(save_dir=args.output_dir)
        losses.append(loss)

        if i % 100 == 0:
            loss_check = report_process(args, i, loss, loss_check, losses)

    export_final_results(args, args.output_dir, losses, base_mesh, text2mesh_model.mlp, network_input, vertices)