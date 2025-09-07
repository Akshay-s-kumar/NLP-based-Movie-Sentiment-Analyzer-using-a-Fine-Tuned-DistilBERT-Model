from zenml import pipeline
from steps.data_loader import load_and_merge_datasets
from steps.cleaner import clean_reviews
from steps.preprocessor import preprocess_for_model
from steps.model_trainer import train_model
from steps.evaluator import evaluate_model

@pipeline
def movie_review_pipeline(learning_rate: float, batch_size: int, epochs: int,dropout: float, optimizer_name: str):
    df = load_and_merge_datasets()
    cleaned_df = clean_reviews(df)
    X_train, X_test, y_train, y_test = preprocess_for_model(cleaned_df)
    model_path = train_model(
        X_train=X_train,
        y_train=y_train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        optimizer_name=optimizer_name
    )
    evaluate_model(model_path=model_path, X_test=X_test, y_test=y_test)