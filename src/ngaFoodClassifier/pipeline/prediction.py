import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.ngaFoodClassifier.components.data_transformation import get_transforms
from src.ngaFoodClassifier.utils.transfer_models import TransferLearningModel
from src.ngaFoodClassifier import logger


class PredictionPipeline:
    def __init__(self,image_path):
        self.image_path =image_path

    
    def predict(self, App=False):
        # Get class labels from the directory structure
        class_labels = ['akara', 'banga soup', 'edikaikong soup',
        'egusi soup', 'ewedu soup', 'jollof rice',
        'masa', 'moimoi', 'ogbono soup', 'okra soup']

        # Load the pre-trained model
        model = TransferLearningModel(
            architecture='resnet50', 
            num_classes=len(class_labels),  # Use number of classes
            device='cpu',
            use_custom_loading=True  
        ).get_model()

        # Load the state dictionary
        model.load_state_dict(torch.load("artifacts/training/model/resnet/resnet50_epoch100_batch16.pt", map_location=torch.device('cpu')))
        model.eval()

        # Define transformations for image preprocessing
        transform = get_transforms(False)

        # Load and preprocess the image
        image = Image.open(self.image_path)
        input_data = transform(image).unsqueeze(0)
        # Add a batch dimension

        # Make predictions
        with torch.no_grad():
            output = model(input_data)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        predicted_label = class_labels[predicted_class]

        if App:
            return str.upper(predicted_label)
        else:
            logger.info(f'Predicted Class: {str.upper(predicted_label)}')

            # Visualize the image and predicted class
            plt.imshow(image)
            plt.title(f'Predicted Class: {str.upper(predicted_label)}')
            plt.axis('off')
            plt.savefig("prediction", dpi=300)
