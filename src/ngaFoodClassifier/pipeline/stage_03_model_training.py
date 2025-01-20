from src.ngaFoodClassifier.config.configuration import ConfigurationManager
from src.ngaFoodClassifier.components.model_training import Training
from src.ngaFoodClassifier import logger


STAGE_NAME = "Data Transformation stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        training_config = config.get_model_training_config()
        training = Training(config=training_config)
        training.train()
