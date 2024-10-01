import os
import urllib.request as request
import zipfile
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size
from src.ngaFoodClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    A class to handle data ingestion tasks including downloading and extracting files.

    Attributes
    ----------
    config : DataIngestionConfig
        Configuration object containing details like source URL, local file path, 
        and extraction directory.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with a configuration.

        Parameters
        ----------
        config : DataIngestionConfig
            A configuration object containing necessary details for data ingestion,
            including the source URL, local file path, and extraction directory.
        """
        self.config = config
        
    
    def download_file(self) -> None:
        """
        Downloads a file from the source URL if it doesn't exist locally.

        If the file already exists, it logs the file's size instead of downloading it again.

        Raises
        ------
        FileNotFoundError
            If the file cannot be downloaded due to a network or file path issue.
        """
        if not os.path.exists(self.config.local_data_file):
            try:
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} downloaded! with following info: \n{headers}")
            except FileNotFoundError as e:
                logger.error(f"Failed to download the file: {e}")
                raise
        else:
            logger.info(f"File already exists with size: {get_size(Path(self.config.local_data_file))}")  

    
    def extract_zip_file(self) -> None:
        """
        Extracts a zip file to a specified directory.

        The directory is created if it does not exist. If the file extraction fails 
        due to a missing or corrupted file, it raises a FileNotFoundError.

        Raises
        ------
        FileNotFoundError
            If the specified zip file does not exist or is corrupted.
        zipfile.BadZipFile
            If the zip file is invalid or corrupted.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        if not os.path.exists(self.config.local_data_file):
            raise FileNotFoundError(f"The file {self.config.local_data_file} does not exist and cannot be extracted.")
        
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract the zip file: {e}")
            raise
