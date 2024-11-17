import os
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
NUM_WORKERS = os.cpu_count()
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
"""
Module-level constants defining file paths for configuration and parameter files.

Attributes
----------
CONFIG_FILE_PATH : pathlib.Path
    The path to the configuration YAML file (`config/config.yaml`).
PARAMS_FILE_PATH : pathlib.Path
    The path to the parameters YAML file (`params.yaml`).
"""
