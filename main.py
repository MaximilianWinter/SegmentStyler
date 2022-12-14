import datetime

from src.helper.paths import LOCAL_MODELS_PATH
from src.helper.helpers import unpickle_data
from src.utils.processing import train
from src.models import *
from src.utils.loss import *
from src.utils.config import get_parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    config = {
        "model": locals()[args.model_name],
        "loss": locals()[args.loss_name],
        "log_dir": f"logs/{str(datetime.date.today())}"
    }

    train(args, config)
