from src.ngaFoodClassifier import logger  # Importing the logger for logging events
from src.ngaFoodClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline  # Importing the data ingestion pipeline

# Define the name of the current stage for logging purposes
STAGE_NAME = "Data Ingestion stage"

try:
    # Log the start of the data ingestion stage
    logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
    
    # Create an instance of the DataIngestionTrainingPipeline
    ingestion = DataIngestionTrainingPipeline()
    
    # Execute the main method of the ingestion pipeline
    ingestion.main()
    
    # Log the completion of the data ingestion stage
    logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception if any error occurs during the process
    logger.exception(e)
    raise e  # Re-raise the exception to propagate it further
