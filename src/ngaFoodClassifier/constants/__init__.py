import os
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
NUM_WORKERS = os.cpu_count()
MEAN = (0.6324, 0.5163, 0.3667)
STD = (0.2613, 0.2721, 0.3229)
