from src.ngaFoodClassifier.config.configuration import ConfigurationManager
from src.ngaFoodClassifier.components.data_transformation import DataTransformation
from src.ngaFoodClassifier import logger


STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_data()
        #data_transformation.extract_zip_file()
