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
    
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    source_dir: str
    source_dir_test: str
    model_dir: str
    model_metrics_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    model_name: str
