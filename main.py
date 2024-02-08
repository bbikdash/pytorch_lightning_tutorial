"""
PyTorch Lightning tutorial modified by Bassam Bikdash
October 2022
"""

import os
from argparse import ArgumentParser
import torch
from torch import utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from model import LitMNIST

# Example modified from: https://github.com/Lightning-AI/tutorials/blob/6d5634b7942180e6ba4a30bfbd74926d1c22f1eb/lightning_examples/mnist-hello-world/hello-world.py

#%% Parse arguments
parser = ArgumentParser()

# add PROGRAM level args
# Program arguments (data_path, cluster_email, etc…)
parser.add_argument_group("Program_Args")
parser.add_argument("--data_dir", type=str, default="./MNIST")

# Add model specific args
# These models will be used in the creation of the model architecture in the constructor of LitMNIST in model.py
parser.add_argument_group("Model_Args")
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--hidden_size", type=int, default=64)

# Add trainer level args
# These arguments are parsed internally by pl.Trainer and correspond to basic attributes of the Trainer class
parser.add_argument_group("Training_Args")
parser.add_argument("-lr", "--learning_rate", type=str, default=5e-5)
parser.add_argument("--max_epochs", type=str, default=10)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("-a", "--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=-1)
args = parser.parse_args()

#%% Initialize the autoencoder with the namespace directly
# model = LitMNIST(args)
model = LitMNIST.load_from_checkpoint("./logs/MNIST Logs/version_2/checkpoints/epoch=9-step=2150.ckpt")

#%% Setup data loading
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)


#%% Train the model
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=3,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="./logs/"),
)

# trainer = Trainer.from_argparse_args(args,
#     callbacks=[TQDMProgressBar(refresh_rate=20)],
#     logger=TensorBoardLogger(save_dir="./logs/", name="MNIST Logs"),
    # logger=CSVLogger(save_dir="./logs/"),
    # default_root_dir="./checkpoints"
# )
trainer.fit(model)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
# trainer = pl.Trainer(accelerator="gpu", devices=1, limit_train_batches=100, max_epochs=1)

#%% Testing
#
# To test a model, call `trainer.test(model)`.
#
# Or, if you've just trained a model, you can just call `trainer.test()` and Lightning will automatically
# test using the best saved checkpoint (conditioned on val_loss).
trainer.test(ckpt_path="best")

#%% Deploy the model
# load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitMNIST.load_from_checkpoint(checkpoint)



