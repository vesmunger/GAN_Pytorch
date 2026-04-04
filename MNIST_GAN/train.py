import pytorch_lightning as pl
from GAN.MNIST_GAN.datamodule import MNISTDataModule
from GAN.MNIST_GAN.gan_module import GAN
import torch

# Setup
dm = MNISTDataModule()
model = GAN()

# GPU detection
AVAIL_GPUS = min(1, torch.cuda.device_count())

trainer = pl.Trainer(max_epochs=20, accelerator='gpu' if AVAIL_GPUS else 'cpu', devices=AVAIL_GPUS or None)
trainer.fit(model, dm)