import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from zenml import step

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_review_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@step
def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['text', 'label'])
    df = df.drop_duplicates(subset=['text'])
    df['clean_text'] = df['text'].apply(clean_review_text)
    return df
