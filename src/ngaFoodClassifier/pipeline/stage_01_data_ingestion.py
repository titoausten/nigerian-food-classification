from src.ngaFoodClassifier.config.configuration import ConfigurationManager  # Importing the configuration manager
from src.ngaFoodClassifier.components.data_ingestion import DataIngestion  # Importing the DataIngestion class
from src.ngaFoodClassifier import logger  # Importing the logger for logging events

STAGE_NAME = "Data Ingestion stage"  # Defining the stage name for logging

class DataIngestionTrainingPipeline:
    """
    Class to manage the data ingestion training pipeline.
    """

    def __init__(self):
        """
        Initializes the DataIngestionTrainingPipeline class.
        Currently, no attributes are defined in the constructor.
        """
        pass

    def main(self):
        """
        Main method to execute the data ingestion process.
        - Loads the data ingestion configuration.
        - Creates a DataIngestion object.
        - Downloads the data file.
        - Extracts the contents of the zip file.
        """
        config = ConfigurationManager()  # Instantiate the configuration manager
        data_ingestion_config = config.get_data_ingestion_config()  # Retrieve data ingestion configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)  # Create a DataIngestion object
        data_ingestion.download_file()  # Download the data file
        data_ingestion.extract_zip_file()  # Extract the downloaded zip file

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")  # Log the start of the data ingestion stage
        ingestion = DataIngestionTrainingPipeline()  # Instantiate the pipeline
        ingestion.main()  # Execute the main data ingestion process
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")  # Log the completion of the stage
    except Exception as e:
        logger.exception(e)  # Log any exceptions that occur
        raise e  # Raise the exception for further handling
