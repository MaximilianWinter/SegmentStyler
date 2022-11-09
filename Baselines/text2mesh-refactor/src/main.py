import pytorch_lightning as pl
from model import Text2Mesh

if __name__ == "__main__":
    trainer = pl.Trainer()
    args = {}
    t2m = Text2Mesh(args)
    trainer.fit(t2m)