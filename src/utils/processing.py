from tqdm import tqdm
import copy
import torch
from pathlib import Path
import numpy as np
import random
import wandb
from pytorch_lightning.utilities.seed import seed_everything

from src.data.mesh import Mesh
from src.utils.Normalization import MeshNormalizer
from src.utils.trainer import Trainer
from src.utils.export import export_final_results
from src.utils.utils import report_process, device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(seed)
    # torch.use_deterministic_algorithms(True)


def train(args, config, wand_proj="dl3d", team="meshers"):

    wandb.init(project=wand_proj, config=args.__dict__, entity=team)

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
    dataset = config["dataset"](args.prompts, args.noisy_masks)
    data_dict = dataset[args.sample]
    dataset.move_batch_to_device(data_dict, device)
    if "labels" in data_dict.keys():
        dataset.visualize_predicted_maps(data_dict["mesh"].vertices.cpu().numpy(), data_dict["labels"], log_path.joinpath("part_map.png"))
        wandb.log({'part_map': wandb.Image(str(log_path.joinpath("part_map.png")))}, step=0)

    base_mesh = data_dict["mesh"]
    MeshNormalizer(base_mesh)()
    text2mesh_model = config["model"](args, data_dict)

    if args.weights_path != "new":
        text2mesh_model.mlp = torch.load(args.weights_path)

    # DEFINE OPTIMIZER
    if args.optimize_gauss_estimator:
        params = text2mesh_model.gauss_estimator.parameters()
    elif args.optimize_learned_labels:
        params = [text2mesh_model.learned_labels]
    else:
        params = text2mesh_model.mlp.parameters()

    optimizer = torch.optim.Adam(
        params, args.learning_rate, weight_decay=args.decay
    )
    activate_scheduler = (
        args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    )
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay_step, gamma=args.lr_decay
        )
    else:
        lr_scheduler = None

    # DEFINE LOSS
    loss_func = config["loss"]

    # NETWORK INPUT
    vertices = copy.deepcopy(base_mesh.vertices)
    network_input = copy.deepcopy(vertices)
    if args.symmetry == True:
        network_input[:, 2] = torch.abs(network_input[:, 2])

    if args.standardize == True:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0)) / torch.std(
            network_input, dim=0
        )

    # TEXT PROMPT
    if not args.prompts:
        raise ValueError("No prompts given.")
    else:
        prompt_path = log_path.joinpath("prompts.txt")
        with open(prompt_path, "w") as fp:
            for prompt in args.prompts:
                fp.write(prompt+"\n")

    log_path.joinpath("sample_id.txt").write_text(str(args.sample))

    trainer = Trainer(
        text2mesh_model, network_input, args.prompts, optimizer, lr_scheduler, loss_func
    )

    losses = []
    loss_check = None
    for i in tqdm(range(args.n_iter)):
        loss_dict = trainer.training_step(
            i, wandb=wandb, clipavg=args.clipavg, save_dir=log_path
        )
        loss = list(loss_dict.values())[
            0
        ]  # This is not really nice. At some point we should adapt report_process, etc.
        losses.append(loss)

        if i % 100 == 0:
            loss_check = report_process(args, i, loss, loss_check, losses)
        
        if trainer.model.stop_loop:
            print("Flag for stopping optimization was set.")
            break


    export_final_results(log_path, losses, text2mesh_model, wandb)

def zip_arrays(left, right):
    rows = []
    for li, ri in zip(left, right):
        row = np.concatenate([li,ri])
        rows.append(row)
    return np.array(rows)