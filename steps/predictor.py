import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from zenml import step
from steps.model_trainer import SimpleDistilBERTClassifier
import mlflow

@step
def predict_review(user_input: str) -> dict:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess(text):
        tokens = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return tokens['input_ids'], tokens['attention_mask']

    input_ids, attention_mask = preprocess(user_input)

    model_paths = {
        "DistilBERT": r"C:\Users\LENOVO\MOVIE_REVIEW_CLASSIFIER\mlruns\models\simple_distilbert_modelBest.pt"
    }

    results = {}

    for model_name, path in model_paths.items():
        model = SimpleDistilBERTClassifier()
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()

            label = "Positive" if pred_label == 1 else "Negative"
            confidence_pct = round(confidence * 100, 2)

            result_str = f"{label} ({confidence_pct}%)"
            print(f" {model_name}: {result_str}")
            results[model_name] = result_str

            # MLflow Logging
            mlflow.log_metric(f"{model_name}_conf", confidence)
            mlflow.log_param(f"{model_name}_label", label)

    print("\n All predictions logged to MLflow.")
    return results
