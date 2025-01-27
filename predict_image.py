from src.ngaFoodClassifier.pipeline.prediction import PredictionPipeline


def main():
    print("Model architectures:\n\
          1 - Efficientnet B0\n\
          2 - Resnet50")
    
    architecture = int(input("\nEnter preferred model architecture: "))
    image_path =  input("Path to image for prediction: ")

    prediction = PredictionPipeline(image_path=image_path, architecture=architecture)
    prediction.predict()


if __name__ == "__main__":
    main()
