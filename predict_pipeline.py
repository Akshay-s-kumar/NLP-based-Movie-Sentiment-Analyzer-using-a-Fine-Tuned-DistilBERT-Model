from zenml import pipeline
from steps.predictor import predict_review  

@pipeline
def prediction_pipeline(user_input: str):
    predict_review(user_input)
