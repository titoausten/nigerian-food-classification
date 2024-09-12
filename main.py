from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    ingestion = DataIngestionTrainingPipeline()
    ingestion.main()
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
