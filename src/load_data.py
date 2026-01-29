import pandas as pd
from .config import REVIEWS_PATH, META_PATH

def load_reviews() -> pd.DataFrame:
    print(f"Loading reviews from {REVIEWS_PATH}")
    df = pd.read_json(REVIEWS_PATH, lines=True)
    print("Reviews shape:", df.shape)
    return df

def load_metadata() -> pd.DataFrame:
    print(f"Loading metadata from {META_PATH}")
    df = pd.read_json(META_PATH, lines=True)
    print("Metadata shape:", df.shape)
    return df
