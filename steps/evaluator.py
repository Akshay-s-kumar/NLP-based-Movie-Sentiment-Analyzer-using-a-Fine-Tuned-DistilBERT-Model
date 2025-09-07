import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from zenml import step
from steps.model_trainer import SimpleDistilBERTClassifier
import mlflow
from tqdm import tqdm
import os

@step(experiment_tracker="mlflow_local_tracker")
def evaluate_model(model_path: str, X_test, y_test):
    model = SimpleDistilBERTClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    y_preds, y_true = [], []

    print("\n🔍 Starting evaluation...\n")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            y_preds.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = accuracy_score(y_true, y_preds)
    clf_report = classification_report(y_true, y_preds, output_dict=False)

    # ✅ Print results
    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:\n", clf_report)

    # ✅ Log to MLflow
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_text(classification_report(y_true, y_preds), "classification_report.txt")

    # 🔸 Optionally save predictions
    os.makedirs("artifacts", exist_ok=True)
    import pandas as pd
    results_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_preds
    })
    results_path = "artifacts/test_predictions.csv"
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)

    print("📄 Predictions saved to artifacts/test_predictions.csv")
