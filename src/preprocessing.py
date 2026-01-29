import re
import pandas as pd

def basic_row_filtering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [c for c in ["user_id", "parent_asin", "text"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)
    if "rating" in df.columns:
        df = df[df["rating"].between(1, 5)]
    return df

_cleaner = re.compile(r"[^a-z0-9\s]+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _cleaner.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["clean_text"] = df["text"].apply(clean_text)
    return df
