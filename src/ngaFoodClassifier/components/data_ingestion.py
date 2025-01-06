import os
import shutil
import urllib.request as request
import zipfile
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size, create_directories
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

    '''
    def extract_zip_file(self) -> None:
        unzip_path = self.config.unzip_dir
        # os.makedirs(unzip_path, exist_ok=True)
        create_directories([unzip_path])
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    '''
    '''
    def extract_zip_file(self) -> None:
        """
        Extracts train and test folders directly into unzip_dir
        """
        unzip_path = self.config.unzip_dir
        create_directories([unzip_path])
        
        # First, extract to a temporary directory
        temp_path = os.path.join(unzip_path, "temp")
        create_directories([temp_path])
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # Move train and test folders to the desired location
        dataset_dir = os.path.join(temp_path, os.listdir(temp_path)[0])  # Get the dataset folder
        
        # Move train and test folders up
        for folder in ['train', 'test']:
            src = os.path.join(dataset_dir, folder)
            dst = os.path.join(unzip_path, folder)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        
        # Clean up temporary directory
        shutil.rmtree(temp_path)
        '''
    def extract_zip_file(self) -> None:
        """
        Extracts train and test folders to unzip_dir regardless of their depth in the zip file
        """
        unzip_path = self.config.unzip_dir
        create_directories([unzip_path])
        
        # First, extract to a temporary directory
        temp_path = os.path.join(unzip_path, "temp")
        create_directories([temp_path])
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(temp_path)

        def find_train_test_dirs(directory):
            """Recursively find train and test directories"""
            train_dir = None
            test_dir = None
            
            for root, dirs, _ in os.walk(directory):
                for dir_name in dirs:
                    if dir_name == 'train' and not train_dir:
                        train_dir = os.path.join(root, dir_name)
                    elif dir_name == 'test' and not test_dir:
                        test_dir = os.path.join(root, dir_name)
                    
                    if train_dir and test_dir:
                        return train_dir, test_dir
            
            return train_dir, test_dir

        # Find train and test directories
        train_src, test_src = find_train_test_dirs(temp_path)
        
        if not train_src or not test_src:
            raise FileNotFoundError("Could not find both train and test directories in the zip file")

        # Move train and test folders to the desired location
        for src, folder in [(train_src, 'train'), (test_src, 'test')]:
            dst = os.path.join(unzip_path, folder)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
        
        # Clean up temporary directory
        shutil.rmtree(temp_path)
