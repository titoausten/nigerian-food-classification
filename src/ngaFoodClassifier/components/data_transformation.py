from src.ngaFoodClassifier.constants import *
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size, create_directories
from src.ngaFoodClassifier.entity.config_entity import DataTransformationConfig
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


class TensorDataset(Dataset):
    def __init__(self, folder_path):
        # Get all the .pt files from the specified folder
        self.file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.pt')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load tensor from the .pt file
        image, label = torch.load(self.file_paths[idx])
        return image, label


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        # Create transforms
        data_transform = transforms.Compose([
            transforms.Resize((224,224)), #CHECK FOR RIGHT FIGURES
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD) #CHECK FOR RIGHT FIGURES
            ])
        
        train_data = datasets.ImageFolder(self.config.train_data_dir, transform=data_transform)
        test_data = datasets.ImageFolder(self.config.test_data_dir, transform=transforms.ToTensor())

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

            torch.save((image, label), image_path)

            if i % 100 == 0:
                logger.info(f'Saved {i} tensors to {image_path} folder')
        

    def create_dataloaders(self):

        # Create custom datasets for loading tensors
        # train_dataset = TensorDataset(self.config.train_tensor_dir)
        # test_dataset = TensorDataset(self.config.test_tensor_dir)

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

        # Get class names
        # class_names = self.transform_data()[0].classes
        # class_names = train_dataset.classes
        class_names = datasets.ImageFolder(self.config.train_data_dir).classes

        return train_dataloader, test_dataloader, class_names
