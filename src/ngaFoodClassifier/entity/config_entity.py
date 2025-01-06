from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    source_dir: str
    train_data_dir: str
    test_data_dir: str
    train_tensor_dir: str
    test_tensor_dir: str
    batch_size: int
