# 🎬 NLP-based Movie Sentiment Analyzer using a Fine-Tuned DistilBERT Model

This project implements a **complete, end-to-end MLOps pipeline** for movie review sentiment analysis.  
It leverages a **fine-tuned DistilBERT model** to classify reviews as **positive** or **negative**.  

The workflow is orchestrated with **ZenML** for reproducibility and modularity, and **MLflow** for comprehensive experiment tracking.  

---

## ✨ Features
- **End-to-End MLOps Pipeline**: Clear, reproducible workflow using ZenML, from data loading to evaluation.  
- **State-of-the-Art Model**: Fine-tuned `distilbert-base-uncased` from Hugging Face Transformers.  
- **Comprehensive Experiment Tracking**: MLflow logs hyperparameters, metrics, and artifacts.  
- **Automated Hyperparameter Tuning**: Scripted experiments with different learning rates, batch sizes, dropouts, and optimizers.  
- **Modular & Reusable Steps**: Separate components for data loading, cleaning, preprocessing, training, and evaluation.  
- **Inference Pipeline**: Supports predictions on new user reviews.  
- **Text Preprocessing**: Dedicated cleaning step (lowercasing, punctuation/digit removal, stopword removal).  

---

## ⚙️ Project Architecture & Pipeline
The main training pipeline **`movie_pipeline.py`** has five sequential steps (via ZenML):

1. **`load_and_merge_datasets`**: Loads Rotten Tomatoes reviews from Hugging Face datasets.  
2. **`clean_reviews`**: Cleans raw reviews for tokenization.  
3. **`preprocess_for_model`**: Tokenizes using `DistilBertTokenizerFast` and splits into train (70%) / test (30%).  
4. **`train_model`**: Fine-tunes `SimpleDistilBERTClassifier`. Logs model + metrics to MLflow.  
5. **`evaluate_model`**: Evaluates test set, prints accuracy + classification report, logs results to MLflow.  

---

## 🛠️ Technologies Used
- **MLOps Framework**: [ZenML](https://zenml.io/)  
- **Experiment Tracking**: [MLflow](https://mlflow.org/)  
- **Deep Learning**: PyTorch, Hugging Face (Transformers, Datasets)  
- **Data Manipulation**: Pandas  
- **Text Processing**: NLTK  
- **Machine Learning**: Scikit-learn  

---
