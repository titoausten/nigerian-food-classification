from src.ngaFoodClassifier.constants import *
import os
from pathlib import Path
from src.ngaFoodClassifier.utils.common import read_yaml, create_directories
from src.ngaFoodClassifier.entity.config_entity import (DataIngestionConfig,
                                                        DataTransformationConfig,
                                                        ModelTrainingConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
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
            train_tensor_dir=config.train_tensor_dir,
            test_tensor_dir=config.test_tensor_dir,
            batch_size=self.params.BATCH_SIZE)
        
        return data_transformation_config
            
     
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            source_dir=config.source_dir,
            source_dir_test=config.source_dir_test,
            model_dir=config.model_dir,
            model_metrics_dir=config.model_metrics_dir,
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.NUM_EPOCHS,
            learning_rate=self.params.LEARNING_RATE,
            model_name=config.model_name
            )

        return model_training_config
