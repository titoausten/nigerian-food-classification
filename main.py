from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.ngaFoodClassifier.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.ngaFoodClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    ingestion = DataIngestionPipeline()
    ingestion.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    transformation = DataTransformationPipeline()
    transformation.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    training = ModelTrainingPipeline()
    training.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
