import torch
from torch import nn
from typing import Dict, List, Tuple, Union
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights
)
from torchvision.models._api import WeightsEnum
from src.ngaFoodClassifier.utils.common import *


class TransferLearningModel:
    """A class to handle transfer learning for various model architectures."""
    
    # Model configurations
    MODEL_CONFIGS = {
        'efficientnet': {
            'model_fn': efficientnet_b0,
            'weights_enum': EfficientNet_B0_Weights,
            'feature_extractor': 'features',
            'classifier': 'classifier',
            'in_features': 1280
        },
        'resnet18': {
            'model_fn': resnet18,
            'weights_enum': ResNet18_Weights,
            'feature_extractor': None,  # ResNet doesn't have a specific feature extractor name
            'classifier': 'fc',
            'in_features': 512
        },
        'resnet50': {
            'model_fn': resnet50,
            'weights_enum': ResNet50_Weights,
            'feature_extractor': None,
            'classifier': 'fc',
            'in_features': 2048
        }
    }

    def __init__(self, 
                 architecture: str,
                 num_classes: int,
                 device: str = "cuda",
                 use_custom_loading: bool = False,
                 dropout_rate: float = 0.2):
        """
        Initialize the transfer learning model.

        Args:
            architecture (str): Model architecture ('efficientnet', 'resnet18', or 'resnet50')
            num_classes (int): Number of output classes
            device (str): Device to use ('cuda' or 'cpu')
            use_custom_loading (bool): Whether to use custom weight loading
            dropout_rate (float): Dropout rate for the classifier
        """
        self.architecture = architecture.lower()
        if self.architecture not in self.MODEL_CONFIGS:
            raise ValueError(f"Architecture must be one of {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[self.architecture]
        self.num_classes = num_classes
        self.device = device
        self.use_custom_loading = use_custom_loading
        self.dropout_rate = dropout_rate
        
        # Build and prepare the model
        self.model, self.preprocess = self._build_model()
        
    def _build_model(self) -> Tuple[nn.Module, callable]:
        """Build and prepare the model for transfer learning."""
        # Handle weight loading based on approach
        if self.use_custom_loading:
            WeightsEnum.get_state_dict = get_state_dict
            weights = self.config['weights_enum'].IMAGENET1K_V1
            model = self.config['model_fn'](weights=weights).to(self.device)
            preprocess = None
        else:
            weights = self.config['weights_enum'].DEFAULT
            model = self.config['model_fn'](weights=weights).to(self.device)
            preprocess = weights.transforms()

        # Freeze feature extraction layers
        self._freeze_features(model)
        
        # Replace classifier
        self._replace_classifier(model)
        
        return model, preprocess
    
    def _freeze_features(self, model: nn.Module):
        """Freeze all layers except the classifier."""
        if self.config['feature_extractor']:
            # For models like EfficientNet with explicit feature extractor
            for param in getattr(model, self.config['feature_extractor']).parameters():
                param.requires_grad = False
        else:
            # For ResNet models
            for name, param in model.named_parameters():
                if self.config['classifier'] not in name:
                    param.requires_grad = False
    
    def _replace_classifier(self, model: nn.Module):
        """Replace the classifier layer with a new one."""
        classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=self.config['in_features'], 
                     out_features=self.num_classes)
        ).to(self.device)
        
        setattr(model, self.config['classifier'], classifier)
    
    def get_model(self) -> Union[nn.Module, Tuple[nn.Module, callable]]:
        """
        Get the prepared model and preprocessing transforms.
        
        Returns:
            If using custom loading: model only
            If using standard loading: tuple of (model, preprocessing_transforms)
        """
        if self.use_custom_loading:
            return self.model
        return self.model, self.preprocess
    
    def get_preprocessing(self) -> callable:
        """Get the preprocessing transforms (only for standard loading)."""
        if self.use_custom_loading:
            raise ValueError("Preprocessing transforms not available with custom loading")
        return self.preprocess
'''
# Example usage
def example_usage():
    #"""Example of how to use the TransferLearningModel class."""
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Original approach with custom loading
    efficient_orig = TransferLearningModel(
        architecture='efficientnet',
        num_classes=num_classes,
        device=device,
        use_custom_loading=True
    )
    model_efficient_orig = efficient_orig.get_model()
    
    # Standard PyTorch approach
    resnet50_std = TransferLearningModel(
        architecture='resnet50',
        num_classes=num_classes,
        device=device,
        use_custom_loading=False
    )
    model_resnet50_std, preprocess_resnet50 = resnet50_std.get_model()

    # Using different dropout rate
    resnet18_custom = TransferLearningModel(
        architecture='resnet18',
        num_classes=num_classes,
        device=device,
        dropout_rate=0.5  # Custom dropout rate
    )
    model_resnet18, preprocess_resnet18 = resnet18_custom.get_model()
'''
