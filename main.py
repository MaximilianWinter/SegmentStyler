import argparse
import datetime

from src.helper.paths import LOCAL_MODELS_PATH
from src.helper.helpers import unpickle_data
from src.utils.processing import train
from src.models import *
from src.utils.loss import *

def get_parser():
    parser = argparse.ArgumentParser()

    # Most relevant
    parser.add_argument('--obj_path', type=str, default='data/chair_testmesh.obj')
    parser.add_argument('--prompts', action="append")
    parser.add_argument('--output_dir', type=str, default='output/')

    # Standard Hyperparameters
    parser.add_argument('--n_iter', type=int, default=6000)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    # MLP
    parser.add_argument('--sigma', type=float, default=10.0)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--encoding', type=str, default='gaussian')
    parser.add_argument('--colordepth', type=int, default=2)
    parser.add_argument('--normdepth', type=int, default=2)
    parser.add_argument('--normratio', type=float, default=0.1)
    parser.add_argument('--clamp', type=str, default="tanh")
    parser.add_argument('--normclamp', type=str, default="tanh")
    parser.add_argument('--no_pe', dest='pe',
                        default=True, action='store_false')
    parser.add_argument('--exclude', type=int, default=0)

    # Renderer
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--n_augs', type=int, default=0)
    parser.add_argument('--n_normaugs', type=int, default=0) 
    parser.add_argument('--frontview', action='store_true')
    parser.add_argument('--frontview_std', type=float, default=8)
    parser.add_argument('--frontview_center', nargs=2,
                        type=float, default=[0., 0.])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float, default=None)
    
    # CLIP
    parser.add_argument('--clipmodel', type=str, default='ViT-B/32')
    parser.add_argument('--jit', action="store_true")
    parser.add_argument('--mincrop', type=float, default=1)
    parser.add_argument('--maxcrop', type=float, default=1)
    parser.add_argument('--normmincrop', type=float, default=0.1)
    parser.add_argument('--normmaxcrop', type=float, default=0.1)
    parser.add_argument('--cropsteps', type=int, default=0)
    parser.add_argument('--cropforward', action='store_true')

    # Input/Output
    parser.add_argument('--input_normals', default=False, action='store_true')
    parser.add_argument('--symmetry', default=False, action='store_true')
    parser.add_argument('--only_z', default=False, action='store_true')
    parser.add_argument('--standardize', default=False, action='store_true')
    parser.add_argument('--save_render', action="store_true")
    
    # Other
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--geoloss', action="store_true")

    # Custom
    parser.add_argument('--mask_path', type=str, default="data/chair_testmesh_mask.jsonc")
    parser.add_argument('--reg_lambda', type=float, default=0)
    parser.add_argument('--optimize_displacement',
                        default=False, action="store_true")
    parser.add_argument('--model_name', type=str, default="Text2MeshOriginal")
    parser.add_argument('--loss_name', type=str, default="default_loss")
    parser.add_argument('--no_mesh_log', action="store_true", default=False)
    parser.add_argument('--experiment_group', type=str, default="debug")
    parser.add_argument('--weights_path', type=str, default="new")
    parser.add_argument('--use_previous_prediction', action="store_true", default=False)
    parser.add_argument('--use_initial_prediction', action="store_true", default=False)
    parser.add_argument('--round_renderer_gradients', action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    partglot_model = next(unpickle_data(LOCAL_MODELS_PATH / 'partglot.pkl'))
    attn_maps = partglot_model.get_attn_maps()

    config = {
        "model": locals()[args.model_name],
        "loss": locals()[args.loss_name],
        "log_dir": f"logs/{str(datetime.date.today())}"
    }

    train(args, config)
