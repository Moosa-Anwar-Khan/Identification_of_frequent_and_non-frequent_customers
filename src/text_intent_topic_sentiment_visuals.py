import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .config import FIGURES_DIR, TABLES_DIR, RANDOM_STATE


def _ensure_vader() -> None:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def _sentiment_bucket(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    return "Positive" if score >= 0 else "Negative"


def _plot_sentiment_distribution(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    df["sentiment_compound"].dropna().hist(bins=50)
    plt.title("Sentiment (compound) distribution")
    plt.xlabel("compound")
    plt.ylabel("frequency")
    plt.tight_layout()
    path = os.path.join(out_dir, "text_intent_sentiment_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_sentiment_by_label(df: pd.DataFrame, label_col: str, out_dir: str) -> None:
    tmp = df.dropna(subset=["sentiment_compound", label_col])
    if tmp.empty:
        return
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=tmp, x=label_col, y="sentiment_compound")
    plt.title("Sentiment by class")
    plt.tight_layout()
    path = os.path.join(out_dir, "text_intent_sentiment_by_label.png")
    plt.savefig(path, dpi=200)
    plt.close()


def _fit_topics(
    texts: pd.Series,
    n_topics: int,
    max_features: int,
    min_df: int,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=min_df,
        stop_words="english",
        max_features=max_features,
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        learning_method="batch",
    )
    lda.fit(dtm)
    topic_probs = lda.transform(dtm)
    return lda, vectorizer, topic_probs


def _topic_top_words(lda: LatentDirichletAllocation, vectorizer: CountVectorizer, top_n: int) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[:-top_n - 1:-1]
        top_words = ", ".join(feature_names[i] for i in top_idx)
        rows.append(
            {
                "topic": idx,
                "top_words": top_words,
                "topic_label": _general_topic_name(top_words),
            }
        )
    return pd.DataFrame(rows)

def _general_topic_name(top_words: str) -> str:
    s = str(top_words).lower()
    if any(k in s for k in ["nail", "polish", "acrylic", "gel"]):
        return "Nail care"
    if any(k in s for k in ["wig", "bundle"]):
        return "Hair extensions"
    if any(k in s for k in ["shampoo", "conditioner", "scent", "smell"]):
        return "Hair products"
    if any(k in s for k in ["hair", "curl", "straighten"]):
        return "Hair care"
    if any(k in s for k in ["skin", "face", "serum", "moisturizer", "lotion", "cream", "acne"]):
        return "Skincare"
    if any(k in s for k in ["lash", "mascara", "eyeliner", "brow", "eyeshadow"]):
        return "Eye makeup"
    if any(k in s for k in ["brush", "brushes", "sponge", "applicator"]):
        if any(k in s for k in ["cheap", "plastic", "fit"]):
            return "Brush quality"
        if any(k in s for k in ["return", "refund", "money", "waste"]):
            return "Brush value"
        return "Makeup tools"
    if any(k in s for k in ["scent", "fragrance", "perfume", "smell", "soap", "body"]):
        return "Fragrance & body"
    if any(k in s for k in ["price", "money", "worth", "cheap", "expensive", "value"]):
        return "Value & price"
    if any(k in s for k in ["refund", "return", "replacement"]):
        return "Returns"
    if any(k in s for k in ["shipping", "delivery", "package", "seller", "arrived"]):
        return "Shipping"
    if any(k in s for k in ["broken", "broke", "defect", "leak", "disappoint"]):
        return "Product quality"
    if any(k in s for k in ["again", "reorder", "repurchase", "restock"]):
        return "Repurchase intent"
    if any(k in s for k in ["years", "months", "since", "long"]):
        return "Long-term use"
    if any(k in s for k in ["use", "easy", "apply", "work", "works", "used"]):
        return "Usage experience"
    if any(k in s for k in ["love", "great", "like", "amazing"]):
        return "Product satisfaction"
    return "General"


def _topic_name_map(topics_df: pd.DataFrame) -> dict[int, str]:
    mapping = {}
    used = set()
    for _, row in topics_df.iterrows():
        base = str(row.get("topic_label") or "").strip() or "General"
        label = base
        if label in used:
            label = f"{base} (T{int(row['topic'])})"
        used.add(label)
        mapping[int(row["topic"])] = label
    return mapping


def _plot_topic_counts(df: pd.DataFrame, out_dir: str, topic_labels: dict[int, str]) -> None:
    counts = df["dominant_topic"].value_counts().sort_index()
    labels = [topic_labels.get(int(idx), str(idx)) for idx in counts.index]
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=labels, y=counts.values, color="#2a9d8f")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Topic distribution (LDA)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "text_intent_topic_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()

def _plot_topic_vs_label(df: pd.DataFrame, label_col: str, out_dir: str, topic_labels: dict[int, str]) -> None:
    tmp = df.dropna(subset=["dominant_topic", label_col])
    if tmp.empty:
        return
    tmp = tmp.copy()
    tmp["topic_label"] = tmp["dominant_topic"].map(topic_labels)
    ct = pd.crosstab(tmp["topic_label"], tmp[label_col], normalize="columns")
    plt.figure(figsize=(8, 5))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Topic distribution by class")
    plt.xlabel("Label")
    plt.ylabel("Topic")
    plt.tight_layout()
    path = os.path.join(out_dir, "text_intent_topic_vs_label.png")
    plt.savefig(path, dpi=200)
    plt.close()

def _plot_topic_sentiment_overlay(df: pd.DataFrame, out_dir: str, topic_labels: dict[int, str]) -> None:
    tmp = df.dropna(subset=["dominant_topic", "sentiment_bucket"])
    if tmp.empty:
        return
    tmp = tmp.copy()
    tmp["topic_label"] = tmp["dominant_topic"].map(topic_labels)
    counts = (
        tmp.groupby(["topic_label", "sentiment_bucket"]).size().reset_index(name="count")
    )
    counts["proportion"] = counts.groupby("topic_label")["count"].transform(lambda x: x / x.sum())
    pivot = counts.pivot(index="topic_label", columns="sentiment_bucket", values="proportion").fillna(0)
    for col in ["Positive", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["Positive", "Negative"]]

    pivot.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.title("Sentiment overlay by topic")
    plt.xlabel("Topic")
    plt.ylabel("Proportion")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    path = os.path.join(out_dir, "text_intent_topic_sentiment_overlay.png")
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sentiment, topic modeling, and overlay visuals for the text-intent dataset."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=os.path.join("labeled_data", "labeled_data.csv"),
        help="Path to labeled CSV (default: labeled_data/labeled_data.csv).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label_intent",
        help="Label column to use (default: label_intent).",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="clean_text",
        help="Text column to use (default: clean_text; fallback to text if missing).",
    )
    parser.add_argument("--n_topics", type=int, default=10, help="Number of LDA topics (default: 10).")
    parser.add_argument("--top_words", type=int, default=12, help="Top words per topic (default: 12).")
    parser.add_argument("--max_features", type=int, default=3000, help="Max vocabulary size (default: 3000).")
    parser.add_argument("--min_df", type=int, default=10, help="Min document frequency (default: 10).")
    args = parser.parse_args()

    if not os.path.exists(args.data_csv):
        raise SystemExit(f"Missing data file: {args.data_csv}")

    df = pd.read_csv(args.data_csv, low_memory=False)
    text_col = args.text_col if args.text_col in df.columns else "text"
    if text_col not in df.columns:
        raise SystemExit(f"Missing text column: {args.text_col}")
    if args.label_col not in df.columns:
        raise SystemExit(f"Missing label column: {args.label_col}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Sentiment
    _ensure_vader()
    sia = SentimentIntensityAnalyzer()
    df["sentiment_compound"] = df[text_col].fillna("").astype(str).apply(
        lambda t: sia.polarity_scores(t)["compound"] if t.strip() else np.nan
    )
    df["sentiment_bucket"] = df["sentiment_compound"].apply(_sentiment_bucket)

    _plot_sentiment_distribution(df, FIGURES_DIR)
    _plot_sentiment_by_label(df, args.label_col, FIGURES_DIR)

    # Topic modeling
    texts = df[text_col].fillna("").astype(str)
    lda, vectorizer, topic_probs = _fit_topics(
        texts=texts,
        n_topics=args.n_topics,
        max_features=args.max_features,
        min_df=args.min_df,
    )
    df["dominant_topic"] = topic_probs.argmax(axis=1)

    topics_df = _topic_top_words(lda, vectorizer, args.top_words)
    topics_df.to_csv(os.path.join(TABLES_DIR, "text_intent_topics.csv"), index=False)
    topic_labels = _topic_name_map(topics_df)

    _plot_topic_counts(df, FIGURES_DIR, topic_labels)
    _plot_topic_vs_label(df, args.label_col, FIGURES_DIR, topic_labels)
    _plot_topic_sentiment_overlay(df, FIGURES_DIR, topic_labels)

    print("Saved visuals to", FIGURES_DIR)
    print("Saved topics to", os.path.join(TABLES_DIR, "text_intent_topics.csv"))


if __name__ == "__main__":
    main()
