import hydra
import torch
import os.path as osp

@hydra.main(config_path="configs", config_name="attn_maps.yaml")
def get_attn_maps(config):
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
    
    return model.get_attn_maps()

if __name__ == "__main__":
    get_attn_maps()




