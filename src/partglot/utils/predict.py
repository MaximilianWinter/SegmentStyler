import torch

from src.partglot.datamodules.partglot_datamodule import PartglotDataModule
from src.partglot.models.pn_agnostic import PNAgnostic


def get_loaded_model(data_dir, model_path="checkpoints/pn_agnostic.ckpt", batch_size=1):
    datamodule = PartglotDataModule(batch_size=batch_size,
        only_correct=True,
        only_easy_context=False,
        max_seq_len=33,
        only_one_part_name=True,
        seed = 12345678,
        split_sizes = [0.8, 0.1, 0.1],
        balance=True,
        data_dir=data_dir)

    model = PNAgnostic(text_dim=64,
            embedding_dim=100,
            sup_segs_dim=64,
            lr=1e-3,
            data_dir=data_dir,
            word2int=datamodule.word2int,
            total_steps=1,
            measure_iou_every_epoch=True,
            save_pred_label_every_epoch=False)

    ckpt = torch.load(model_path)
    if "state_dict" in ckpt:
        print("write state dict")
        ckpt = ckpt["state_dict"]

    model.load_state_dict(ckpt)
    
    return model, datamodule