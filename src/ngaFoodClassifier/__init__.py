import os  # Importing the os module for operating system interactions
import sys  # Importing the sys module for system-specific parameters and functions
import logging  # Importing the logging module for logging events

# Define the format for the logging messages
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Specify the directory for log files
log_dir = "logs"
# Create the full path for the log file
log_filepath = os.path.join(log_dir, "running_logs.log")
# Create the log directory if it doesn't already exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Set the logging format

    handlers=[  # Specify the handlers for logging
        logging.FileHandler(log_filepath),  # Log messages to a file
        logging.StreamHandler(sys.stdout)    # Log messages to standard output
    ]
)

# Create a logger object with the specified name
logger = logging.getLogger("ngaClassifierLogger")
