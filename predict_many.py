
from src.ngaFoodClassifier.pipeline.prediction import PredictionPipeline
import os

def main():
    print("Model architectures:\n\
          1 - Efficientnet B0\n\
          2 - Resnet50")
    
    architecture = int(input("\nEnter preferred model architecture: "))
    image_path =  "test_images"

    for file in os.listdir(image_path):
        file = os.path.join(image_path, file)
        prediction = PredictionPipeline(image_path=file, architecture=architecture)
        prediction.predict()


if __name__ == "__main__":
    main()


