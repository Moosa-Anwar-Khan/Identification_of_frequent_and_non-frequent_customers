import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from .config import FIGURES_DIR, PROJECT_ROOT


def _word_count(text: str) -> int:
    return len(str(text).split())


def _safe_label(label: str) -> str:
    return str(label).strip() if label is not None else "Unknown"


def _plot_class_balance(df: pd.DataFrame, label_col: str, out_dir: str) -> None:
    counts = df[label_col].value_counts()
    plot_df = counts.reset_index()
    plot_df.columns = [label_col, "count"]
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=plot_df,
        x=label_col,
        y="count",
        hue=label_col,
        legend=False,
        palette="muted",
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Class balance (text intent)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "text_intent_class_balance.png"), dpi=200)
    plt.close()


def _plot_length_distribution(
    df: pd.DataFrame, label_col: str, text_col: str, out_dir: str
) -> None:
    tmp = df[[label_col, text_col]].copy()
    tmp["word_count"] = tmp[text_col].apply(_word_count)
    cutoff = tmp["word_count"].quantile(0.98)
    tmp = tmp[tmp["word_count"] <= cutoff]
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=tmp,
        x="word_count",
        hue=label_col,
        bins=40,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.xlabel("Review length (words)")
    plt.ylabel("Density")
    plt.title("Review length distribution by class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "text_intent_length_distribution.png"), dpi=200)
    plt.close()


def _plot_top_terms(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    out_dir: str,
    top_n: int,
    max_features: int,
) -> None:
    labels = sorted(df[label_col].dropna().unique())
    if len(labels) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, label in zip(axes, labels[:2]):
        subset = df[df[label_col] == label]
        texts = subset[text_col].astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
        X = vectorizer.fit_transform(texts)
        counts = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_idx = counts.argsort()[-top_n:][::-1]
        top_terms = [terms[i] for i in top_idx]
        top_counts = [counts[i] for i in top_idx]
        ax.barh(top_terms[::-1], top_counts[::-1], color="#2a9d8f")
        ax.set_title(f"Top terms: {_safe_label(label)}")
        ax.set_xlabel("Term frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "text_intent_top_terms.png"), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation visuals for the text-intent dataset."
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "labeled_data", "labeled_data.csv"),
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
    parser.add_argument(
        "--top_n",
        type=int,
        default=12,
        help="Top terms per class (default: 12).",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=2000,
        help="Max vocabulary size (default: 2000).",
    )
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

    _plot_class_balance(df, args.label_col, FIGURES_DIR)
    _plot_length_distribution(df, args.label_col, text_col, FIGURES_DIR)
    _plot_top_terms(
        df,
        args.label_col,
        text_col,
        FIGURES_DIR,
        top_n=args.top_n,
        max_features=args.max_features,
    )

    print("Saved visuals to", FIGURES_DIR)


if __name__ == "__main__":
    main()
