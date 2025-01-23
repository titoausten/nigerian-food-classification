from src.ngaFoodClassifier.pipeline.prediction import PredictionPipeline

image_path =  input("Path to image for prediction: ")
prediction = PredictionPipeline(image_path=image_path)
prediction.predict()
