from tqdm import tqdm
import copy
import torch
from pathlib import Path
import numpy as np
import random
import wandb

from src.data.mesh import Mesh
from src.utils.Normalization import MeshNormalizer
from src.utils.trainer import Trainer
from src.utils.export import export_final_results
from src.utils.utils import report_process

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(args, config, wand_proj='dl3d', team='meshers'):
    
    wandb.init(project=wand_proj,
               config=args.__dict__,
               entity=team)
    
    log_path_base = Path(config["log_dir"]).joinpath(args.output_dir)

    # CREATE OUTPUT DIR
    created_directory = False
    i = 0
    while not created_directory:
        log_path = log_path_base.joinpath(f"version_{i}")
        try:
            log_path.mkdir(parents=True, exist_ok=False)
            created_directory = True
            print(f"Successfully created log directory at {log_path}.")
        except FileExistsError:
            created_directory = False
            i += 1

    # SET SEED
    set_seed(args.seed)

    # LOAD MESH AND MODEL
    base_mesh = Mesh(args.obj_path)
    MeshNormalizer(base_mesh)()
    text2mesh_model = config["model"](args, base_mesh)

    # DEFINE OPTIMIZER
    optimizer = torch.optim.Adam(text2mesh_model.mlp.parameters(), args.learning_rate, weight_decay=args.decay)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)
    else:
        lr_scheduler = None
    
    # DEFINE LOSS
    loss_func = config["loss"]

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
        loss_dict = trainer.training_step(i, wandb=wandb, clipavg=args.clipavg, save_dir=log_path)
        loss = list(loss_dict.values())[0] # This is not really nice. At some point we should adapt report_process, etc.
        losses.append(loss)

        if i % 100 == 0:
            loss_check = report_process(args, i, loss, loss_check, losses)

    export_final_results(args, log_path, losses, base_mesh, text2mesh_model.mlp, network_input, vertices, wandb)