## torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from UNet_lightning import UNet_model

##
from datetime import timedelta, datetime

input_shape = (1, 28, 28)
model = UNet_model(input_shape=input_shape)

checkpoint2 = ModelCheckpoint(
    save_top_k=5,
    monitor="step",
    mode="max",
    train_time_interval=timedelta(minutes=120),
    dirpath='model_weight/v1_fasion',
    filename="DDPM-ep{epoch:03d}-step{step}"
)

trainer = pl.Trainer(gpus=1,
                     max_epochs=20,
                     #max_steps=10,
                     enable_progress_bar=False,
                     callbacks=[checkpoint2],
                     log_every_n_steps=4,
                     #overfit_batches=1
                     )

trainer.fit(model,
            #ckpt_path="model_weight/v1/DDPM-epepoch=007-stepstep=14511.ckpt",
            )

trainer.save_checkpoint("model_weight/v1_fasion/final_model.ckpt")
