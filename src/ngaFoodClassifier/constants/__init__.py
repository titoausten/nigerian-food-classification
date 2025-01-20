import os
import torch
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
NUM_WORKERS = 0
MEAN = (0.6324, 0.5163, 0.3667)
STD = (0.2613, 0.2721, 0.3229)
LOSS_FN= torch.nn.CrossEntropyLoss()
NUM_CLASSES = 10
