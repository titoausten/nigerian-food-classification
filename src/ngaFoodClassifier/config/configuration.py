from src.ngaFoodClassifier.constants import *
from src.ngaFoodClassifier.utils.common import read_yaml, create_directories
from src.ngaFoodClassifier.entity.config_entity import (DataIngestionConfig,
                                                        DataTransformationConfig)


class ConfigurationManager:
    """
    Handles the reading of configuration files and the setup of necessary directories for data ingestion.

    Attributes
    ----------
    config : dict
        The configuration settings read from the YAML file.
    """

    def __init__(self, config_filepath: str = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH)):
        """
        Initializes the ConfigurationManager with the specified configuration file path.
        
        It reads the YAML configuration file and creates necessary directories as specified in the configuration.

        Parameters
        ----------
        config_filepath : str, optional
            The file path to the configuration YAML file, by default `CONFIG_FILE_PATH`.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Prepares and returns the data ingestion configuration based on the YAML configuration.

        This method creates the root directory for data ingestion and initializes a `DataIngestionConfig` object.

        Returns
        -------
        DataIngestionConfig
            A configuration object that contains the necessary paths and URLs for data ingestion.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config


    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            train_data_dir=config.train_data_dir,
            test_data_dir=config.test_data_dir,
            batch_size=self.params.BATCH_SIZE
        )

        return data_transformation_config
