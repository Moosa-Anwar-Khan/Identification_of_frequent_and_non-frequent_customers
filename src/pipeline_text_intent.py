import argparse
import os
from pathlib import Path
from shutil import copy2

import pandas as pd

from .bert_training import run_bert_training
from .config import FIGURES_DIR, ML_DIR, PROJECT_ROOT, TABLES_DIR


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy2(src, dest)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train text-intent BERT from labeled_data.csv and export figures/tables."
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
        help="Label column to train for.",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="bert_text_intent_k5_v5",
        help="Output subdir under outputs/ml/.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_csv)
    if not data_path.exists():
        raise SystemExit(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    run_bert_training(df, label_col=args.label_col, out_subdir=args.out_subdir)

    ml_dir = Path(ML_DIR) / args.out_subdir
    if not ml_dir.exists():
        raise SystemExit(f"Missing model outputs at {ml_dir}")

    figures_dir = Path(FIGURES_DIR)
    tables_dir = Path(TABLES_DIR)

    _copy_if_exists(
        ml_dir / "bert_confusion_matrix_threshold_best.png",
        figures_dir / "bert_confusion_matrix_threshold_best.png",
    )
    _copy_if_exists(
        ml_dir / "bert_precision_recall_curve.png",
        figures_dir / "bert_precision_recall_curve.png",
    )
    _copy_if_exists(
        ml_dir / "bert_threshold_best_classification_report.csv",
        tables_dir / "bert_text_intent_classification_report.csv",
    )
    _copy_if_exists(
        ml_dir / "bert_threshold_best_classification_report.txt",
        tables_dir / "bert_text_intent_classification_report.txt",
    )
    _copy_if_exists(
        ml_dir / "bert_best_threshold.txt",
        tables_dir / "bert_text_intent_best_threshold.txt",
    )

    print("Pipeline complete.")
    print(f"Model outputs: {ml_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()
