from zenml import step
from datasets import load_dataset
import pandas as pd

@step
def load_and_merge_datasets() -> pd.DataFrame:
    imdb = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    df = pd.DataFrame(imdb['train'])
    df['label'] = df['label'].apply(lambda x: 'good' if x == 1 else 'bad')
    df['movie_title'] = 'Unknown_IMDB'
    return df