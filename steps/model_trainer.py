from transformers import DistilBertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from zenml import step
import mlflow
import os
from tqdm import tqdm
import pandas as pd

class SimpleDistilBERTClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.bert_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.bert_layer.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]
        x = self.dropout_layer(cls_output)
        return self.output_layer(x)

@step(experiment_tracker="mlflow_local_tracker")
def train_model(X_train, y_train,
                learning_rate: float = 2e-5,
                batch_size: int = 16,
                epochs: int = 3,
                dropout: float = 0.3,
                optimizer_name: str = "Adam") -> str:

    model = SimpleDistilBERTClassifier(dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mlflow.log_params({
        "model": "DistilBERT",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "device": str(device),
        "dropout": dropout,
        "optimizer": optimizer_name
    })

    train_dataset = TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    batch_losses = {}

    model.train()
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0
        epoch_batch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(progress_bar):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_batch_losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(batch_loss=loss.item())


        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100

        # 🔸 Store per-batch losses
        batch_losses[f"epoch_{epoch+1}"] = epoch_batch_losses

        print(f"✅ Epoch {epoch+1} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("epoch_accuracy", accuracy, step=epoch)

    # 🔸 Save per-batch loss to JSON
        os.makedirs("artifacts", exist_ok=True)
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in batch_losses.items()]))
        loss_csv_path = "artifacts/batch_losses.csv"
        loss_df.to_csv(loss_csv_path, index_label="batch_index")
        mlflow.log_artifact(loss_csv_path)

    # 🔸 Save and log model
    model_path = "artifacts/simple_distilbert_model.pt"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    print("📂 Model and batch losses saved in 'artifacts/' folder")
    return model_path
