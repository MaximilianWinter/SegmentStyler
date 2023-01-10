import hydra
import torch
import os.path as osp
from partglot.utils.simple_utils import pickle_data
from src.helper.paths import LOCAL_MODELS_PATH

@hydra.main(config_path="./configs", config_name="predict.yaml")
def pickle_model(config):
    datamodule = hydra.utils.instantiate(config.datamodule)
    
    model = hydra.utils.instantiate(
        config.model,
        word2int=datamodule.word2int,
        total_steps=1
    )
    ckpt = torch.load(osp.join(config.work_dir, config.ckpt_path))
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(ckpt)
    
    pickle_data(LOCAL_MODELS_PATH / 'partglot.pkl', model)
    
    
if __name__ == "__main__":
    pickle_model()
