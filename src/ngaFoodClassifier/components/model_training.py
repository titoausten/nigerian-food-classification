import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple
from src.ngaFoodClassifier.constants import *
from src.ngaFoodClassifier import logger
from src.ngaFoodClassifier.utils.common import get_size, create_directories
from src.ngaFoodClassifier.entity.config_entity import ModelTrainingConfig
from src.ngaFoodClassifier.utils.transfer_models import TransferLearningModel
from src.ngaFoodClassifier.constants import *
from torch.utils.data import DataLoader, Dataset, Subset


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn,
               optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Perform a single training step for the model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): Data loader for the training data.
        loss_fn: The loss function.
        optimizer: The optimizer.
        device (torch.device): The device on which the training is performed.

    Returns:
        Tuple[float, float]: Tuple containing the average training loss and accuracy for the step.
    """
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Calculate the accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    """
    Perform a single testing step for the model.

    Args:
        model: The PyTorch model to be tested.
        dataloader: Data loader for the testing data.
        loss_fn: The loss function.
        device (torch.device): The device on which the testing is performed.

    Returns:
        Tuple[float, float]: Tuple containing the average testing loss and accuracy for the step.
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            loss = loss_fn(test_pred, y).item()
            test_loss += loss

            test_pred_labels = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


class TensorDataset(Dataset):
    def __init__(self, folder: str):
        self.folder = folder
        self.image_paths = []
        self.labels = []
        self.classes = []
        
        # Get class names (directories)
        self.classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build file list
        for class_name in self.classes:
            class_folder = os.path.join(folder, class_name)
            if os.path.isdir(class_folder):
                for file_name in os.listdir(class_folder):
                    if file_name.endswith('.pt'):
                        self.image_paths.append(os.path.join(class_folder, file_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load tensor with memory efficiency
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Load tensor directly to the correct device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image, _ = torch.load(image_path, map_location=device)
            
            # Ensure tensor is in correct format and memory-efficient
            image = image.contiguous()
            
            # Free unnecessary memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading tensor from {image_path}: {str(e)}")
            raise


class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.checkpoint_frequency = 50  # Save model and loss curves every 50 epochs

    def train(self):
        """
        Train a PyTorch model and evaluate its performance over multiple epochs.

        Args:
            model: The PyTorch model to be trained.
            train_dataloader: Data loader for the training data.
            test_dataloader: Data loader for the testing data.
            optimizer: The optimizer.
            loss_fn: The loss function.
            epochs (int): Number of training epochs.
            device (torch.device): The device on which the training and testing are performed.
        Returns:
            Dict: A dictionary containing training and testing loss and accuracy results.
        """
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Add gradient scaler for mixed precision training
        scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None


        # Dataloaders
        train_dataloader, test_dataloader, class_names = self.create_dataloaders()

        num_classes = len(class_names)

        # Model loading
        model = TransferLearningModel(
            architecture=self.config.model_name,
            num_classes=num_classes,
            device=DEVICE,
            use_custom_loading=True
        ).get_model()

        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []}
        
        torch.cuda.manual_seed(42)
        torch.manual_seed(42)

        optimizer = torch.optim.Adam(model.parameters(), self.config.learning_rate)

        for epoch in range(self.config.epochs):
            # Training step with memory optimization
            with torch.amp.autocast(device_type='cuda'):  # Enable automatic mixed precision
                train_loss, train_acc = train_step(model, train_dataloader,
                                                   loss_fn=LOSS_FN, optimizer=optimizer,
                                                   device=DEVICE)
            
            # Memory cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_loss, train_acc = train_step(model, train_dataloader, 
                                               loss_fn=LOSS_FN, optimizer=optimizer,
                                               device=DEVICE)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn=LOSS_FN,
                                            device=DEVICE)

            logger.info(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Save checkpoint every 50 epochs
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_model(model, epoch=epoch + 1)
                logger.info(f"Checkpoint saved at epoch {epoch + 1}")

                self.plot_loss_curves(results, epoch=epoch + 1)
                logger.info(f"Loss curves saved at epoch {epoch + 1}")
            

        # Save model
        self.save_model(model)
        # Plot loss curves
        self.plot_loss_curves(results)

        # Memory cleanup after each epoch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return results
    

    def save_model(self, model, epoch=None):
    #def save_model(self, model):
        """
        Save a PyTorch model to a specified directory with a given model name.

        Args:
            model: The PyTorch model to be saved.
            target_dir (str): The directory where the model will be saved.
            model_name (str): The name of the model file (should end with '.pt' or '.pth').

        Returns:
            None
        """
        
        create_directories([self.config.model_dir])

        '''
        model_save_path = os.path.join(self.config.model_dir, self.config.model_name + ".pt")

        # If the file already exists, skip or handle differently
        if os.path.exists(model_save_path):
            logger.info(f"File {model_save_path} already exists, skipping.")
        '''
        # Create model name with epoch and batch size information
        if epoch is not None:
            model_filename = f"{self.config.model_name}_epoch{epoch}_batch{self.config.batch_size}.pt"
        else:
            model_filename = f"{self.config.model_name}_main{self.config.epochs}_final_batch{self.config.batch_size}.pt"

        model_save_path = os.path.join(self.config.model_dir, model_filename)

        # If the file already exists, skip or handle differently
        if os.path.exists(model_save_path):
            logger.info(f"File {model_save_path} already exists, creating new version.")
            model_filename = f"{self.config.model_name}_epoch{epoch}_batch{self.config.batch_size}_v2.pt"
            model_save_path = os.path.join(self.config.model_dir, model_filename)

        logger.info(f"Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
        logger.info(f'Model saved to {self.config.model_dir} folder')


    def plot_loss_curves(self, results, epoch=None):
        """
        Plot training and testing loss and accuracy curves.

        Args:
            results (dict): A dictionary containing 'train_loss', 'test_loss', 'train_acc', and 'test_acc' lists.

        Returns:
            None
        """
        data = results

        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # Plot training loss and testing loss
        axes[0].plot(data['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(data['test_loss'], label='Test Loss', marker='o')
        axes[0].set_title('Training and Testing Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        # Plot training accuracy and testing accuracy
        axes[1].plot(data['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(data['test_acc'], label='Test Accuracy', marker='o')
        axes[1].set_title('Training and Testing Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        # Create metrics save path
        metrics_save_path = os.path.join(self.config.model_metrics_dir, self.config.model_name)
        create_directories([metrics_save_path])  # Ensure directory exists


        # Create model name with epoch and batch size information
        if epoch is not None:
            file_name = f"{self.config.model_name}_epoch{epoch}_batch{self.config.batch_size}_training_curves.png"
        else:
            file_name = f"{self.config.model_name}_main{self.config.epochs}_batch{self.config.batch_size}_training_curves.png"


        # Construct full path for the plot file
        full_save_path = os.path.join(metrics_save_path, file_name)

        # Save the plot
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=300)
        logger.info(f"Loss curves saved to {full_save_path}")

        # Free memory by closing the figure
        plt.close(fig)


    def create_dataloaders(self):
        try:
            train_data = TensorDataset(self.config.source_dir)
            test_data = TensorDataset(self.config.source_dir_test)
            class_names = train_data.classes

            logger.info(f"Number of training samples: {len(train_data)}")
            logger.info(f"Number of test samples: {len(test_data)}")
            logger.info(f"Classes found: {class_names}")

            # Verify that train and test have the same classes
            if set(train_data.classes) != set(test_data.classes):
                logger.warning("Train and test datasets have different classes!")
                logger.warning(f"Train classes: {class_names}")
                logger.warning(f"Test classes: {test_data.classes}")

            #train_subset = Subset(train_data, range(100))  # Load the first 100 samples
            #test_subset = Subset(test_data, range(100))

            
            # Reduce memory usage by adjusting DataLoader parameters
            train_dataloader = DataLoader(
                train_data,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=NUM_WORKERS,  # Reduce number of workers
                pin_memory=True,  # False Disable pin_memory to reduce memory usage
                persistent_workers=False,  # Keep workers alive between iterations
                prefetch_factor=None  # Reduce prefetch factor
            )

            test_dataloader = DataLoader(
                test_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,  # Reduce number of workers
                pin_memory=True,  # Disable pin_memory to reduce memory usage
                persistent_workers=False,  # Keep workers alive between iterations
                prefetch_factor=None  # Reduce prefetch factor
            )

            return train_dataloader, test_dataloader, class_names
        
        except Exception as e:
            logger.error(f"Error creating dataloaders: {str(e)}")
            raise
