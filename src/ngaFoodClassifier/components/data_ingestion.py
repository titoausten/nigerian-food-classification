import os
import urllib.request as request
import zipfile
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size
from src.ngaFoodClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    
    def download_file(self) -> None:
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists with size: {get_size(Path(self.config.local_data_file))}")  

    
    def extract_zip_file(self) -> None:
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
