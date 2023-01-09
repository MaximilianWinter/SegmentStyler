import datetime

from src.utils.processing import train
from src.utils.config import get_parser
from src.models import *
from src.utils.loss import *


if __name__ == "__main__":
    args = get_parser().parse_args()

    config = {
        "model": locals()[args.model_name],
        "loss": locals()[args.loss_name],
        "log_dir": f"logs/{str(datetime.date.today())}"
    }

    train(args, config)
