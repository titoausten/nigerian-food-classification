from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for data ingestion settings.
    
    Attributes:
    -----------
    root_dir : Path
        The root directory where data files are stored.
    source_URL : str
        The URL from which to download the data file.
    local_data_file : Path
        The local path for saving the downloaded file.
    unzip_dir : Path
        The directory where the contents of the zip file will be extracted.
    """
    
    root_dir: Path              # Directory to store all data-related files
    source_URL: str             # URL to download the data file from
    local_data_file: Path       # Path where the downloaded file will be saved
    unzip_dir: Path             # Path where the zip file's contents will be extracted


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    source_dir: Path
    train_data_dir: Path
    test_data_dir: Path
    train_tensor_dir: str
    test_tensor_dir: str
    batch_size: Path
