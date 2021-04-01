import os
import yaml
import torch

from model_training.common.trainer import Trainer
from model_training.common.augmentations import get_transforms
from model_training.common.datasets import ImageNetMLC, ImageNetHuman

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(os.path.join(os.path.dirname(__file__), "config", "imagenet.yaml")) as config_file:
    config = yaml.full_load(config_file)
    pass

train_transform = get_transforms(config["train"]["transform"])
val_transform = get_transforms(config["val"]["transform"])

train_ds = ImageNetHuman(config["train"]["path"], transform=train_transform)
val_ds = ImageNetHuman(config["val"]["path"], transform=val_transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"],
                                       shuffle=True, num_workers=8, pin_memory=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"],
                                     shuffle=True, num_workers=8, pin_memory=True)

print("Train: ", len(train_dl), "  Val: ", len(val_dl))

trainer = Trainer(config, train_dl, val_dl)
trainer.train()
