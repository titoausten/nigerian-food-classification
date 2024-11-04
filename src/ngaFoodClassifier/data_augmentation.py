from torchvision.datasets import ImageFolder
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
from PIL import Image
import os

#Definition of transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip the image horizontally with a probability of 0.5
    transforms.RandomRotation(degrees=10),       # Rotate the image by up to 10 degrees
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # Random crop and resize to 32x32
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color changes
    transforms.ToTensor(),                       # Convert to PyTorch tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to have mean 0.5 and std 0.5
])

# Splitting datasets into train and test sets
# Load all data from a single folder
full_dataset = ImageFolder(root='location_of_dataset', transform=transforms.None)

# Define split sizes
train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size # 20% for testing

# Randomly split the dataset
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Load the dataset from the directory and apply transformations
train_dataset = ImageFolder(root='location_of_train_set', transform=transform)
test_dataset = ImageFolder(root='location_of_test_set', transform=transforms.ToTensor())  # Usually, no augmentation for the test set

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
