import os   # Importing the os module for interacting with the operating system
from box.exceptions import BoxValueError  # Importing BoxValueError for handling exceptions from the Box library
import yaml  # Importing yaml for reading YAML files
from src.ngaFoodClassifier import logger  # Importing the logger for logging messages
import json  # Importing json for reading and writing JSON data
import joblib  # Importing joblib for saving and loading binary data
from ensure import ensure_annotations  # Importing ensure_annotations for type checking
from box import ConfigBox  # Importing ConfigBox for structured configuration handling
from pathlib import Path  # Importing Path for path manipulations
from typing import Any  # Importing Any for flexible type hints
import base64  # Importing base64 for encoding and decoding base64 data


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        e: If there is an error opening or reading the file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)  # Load the YAML file content
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")  # Log successful load
            return ConfigBox(content)  # Return content as ConfigBox
    except BoxValueError:
        raise ValueError("yaml file is empty")  # Raise error if file is empty
    except Exception as e:
        raise e  # Raise any other exceptions


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories.

    Args:
        path_to_directories (list): List of paths to create directories.
        verbose (bool, optional): If True, logs the creation of directories. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist
        if verbose:
            logger.info(f"created directory at: {path}")  # Log directory creation


@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data as a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save in the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)  # Write data to JSON file with indentation

    logger.info(f"json file saved at: {path}")  # Log successful save


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as a ConfigBox instead of a dict.
    """
    with open(path) as f:
        content = json.load(f)  # Load content from JSON file

    logger.info(f"json file loaded successfully from: {path}")  # Log successful load
    return ConfigBox(content)  # Return content as ConfigBox


@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as a binary file.

    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to the binary file.
    """
    joblib.dump(value=data, filename=path)  # Save data using joblib
    logger.info(f"binary file saved at: {path}")  # Log successful save


@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: The object stored in the file.
    """
    data = joblib.load(path)  # Load data using joblib
    logger.info(f"binary file loaded from: {path}")  # Log successful load
    return data  # Return loaded data


@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)  # Get size in KB
    return f"~ {size_in_kb}KB"  # Return size formatted as a string


def decodeImage(imgstring, fileName):
    """Decodes a base64 string and saves it as an image file.

    Args:
        imgstring (str): Base64 encoded image string.
        fileName (str): Filename to save the image.
    """
    imgdata = base64.b64decode(imgstring)  # Decode base64 string
    with open(fileName, 'wb') as f:
        f.write(imgdata)  # Write decoded image data to file
        f.close()  # Close file


def encodeImageIntoBase64(croppedImagePath):
    """Encodes an image file into a base64 string.

    Args:
        croppedImagePath (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())  # Return encoded base64 string of the image
