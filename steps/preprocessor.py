from transformers import DistilBertTokenizerFast
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple

@step
def preprocess_for_model(df: pd.DataFrame)-> Tuple[dict, dict, torch.Tensor, torch.Tensor]:
    df['label_encoded'] = df['label'].map({'good': 1, 'bad': 0})
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    tokenized = tokenizer(
        list(df['clean_text']),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt'
    )

    labels = torch.tensor(df['label_encoded'].values)

    input_ids_train, input_ids_test, attn_train, attn_test, y_train, y_test = train_test_split(
        tokenized['input_ids'], tokenized['attention_mask'], labels,
        test_size=0.3, random_state=42, stratify=labels
    )

    X_train = {'input_ids': input_ids_train, 'attention_mask': attn_train}
    X_test = {'input_ids': input_ids_test, 'attention_mask': attn_test}

    return X_train, X_test, y_train, y_test