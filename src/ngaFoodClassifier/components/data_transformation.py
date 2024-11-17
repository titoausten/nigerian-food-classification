from src.ngaFoodClassifier.constants import *
from src.ngaFoodClassifier import logger
#from src.ngaFoodClassifier.utils.common import get_size
from src.ngaFoodClassifier.entity.config_entity import DataTransformationConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        # Create transforms
        data_transform = transforms.Compose([
            transforms.Resize((64, 64)), #CHECK FOR RIGHT FIGURES
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD) #CHECK FOR RIGHT FIGURES
            ])
        
        train_data = datasets.ImageFolder(self.config.train_data_dir, transform=data_transform)
        test_data = datasets.ImageFolder(self.config.test_data_dir, transform=transforms.ToTensor())

        return train_data, test_data


    def create_dataloaders(self):
        # Get class names
        class_names = self.transform_data()[0].classes

        # Create data loaders
        train_dataloader = DataLoader(self.transform_data()[0],
                                    batch_size=self.config.batch_size,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True)

        test_dataloader = DataLoader(self.transform_data()[1],
                                    batch_size=self.config.batch_size,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS,
                                    pin_memory=True)

        return train_dataloader, test_dataloader, class_names
