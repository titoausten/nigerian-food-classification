from src.ngaFoodClassifier.constants import *
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size, create_directories
from src.ngaFoodClassifier.entity.config_entity import DataTransformationConfig
from torchvision import datasets, transforms
import torch


def get_transforms(train=True):
    """Create consistent transforms for both train and test"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        

    def transform_data(self):
        # Create transforms
        train_transform = get_transforms(train=True)
        test_transform = get_transforms(train=False)

        train_data = datasets.ImageFolder(self.config.train_data_dir, transform=train_transform)
        test_data = datasets.ImageFolder(self.config.test_data_dir, transform=test_transform)

        # Save Tensors
        self.save_tensor(train_data, self.config.train_tensor_dir)
        self.save_tensor(test_data, self.config.test_tensor_dir)

        return train_data, test_data


    def save_tensor(self, data, folder: str):
        create_directories([folder])

        '''
        for i, (image, label) in enumerate(data):
            image_path = os.path.join(folder, f'{i}.pt')
            torch.save((image, label), image_path)
        '''

        for i, (image, label) in enumerate(data):
            class_name = data.classes[label]  # Get the class name based on the label
            class_folder = os.path.join(folder, class_name)

            # create_directories([class_folder])  # Create folder for each class
            # Create the class folder only once (outside the loop)

            if not os.path.exists(class_folder):
                create_directories([class_folder])  # Only create the folder if it doesn't exist

            image_path = os.path.join(class_folder, f'{i}.pt')  # Save as .pt file

            # If the file already exists, skip or handle differently
            if os.path.exists(image_path):
                logger.info(f"File {image_path} already exists, skipping.")
                continue

            # Ensure tensor is contiguous and in the correct format
            image = image.contiguous()
            torch.save((image, label), image_path)

            if i % 100 == 0:
                logger.info(f'Saved {i} tensors to {image_path} folder')
