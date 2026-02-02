import argparse
import json
import os
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .preprocessing import clean_text


def _get_device() -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _read_threshold(model_dir: str) -> float:
    """
    Load tuned threshold for class 1 (Frequent).
    Preferred: <model_dir>/best_threshold.txt (single float).
    Fallback:  <parent>/bert_best_threshold.txt (key=value).
    """
    thr_path = os.path.join(model_dir, "best_threshold.txt")
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        try:
            return float(s)
        except Exception:
            pass

    parent = os.path.dirname(model_dir.rstrip("/\\"))
    fallback = os.path.join(parent, "bert_best_threshold.txt")
    if os.path.exists(fallback):
        with open(fallback, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("best_threshold="):
                    return float(line.split("=", 1)[1].strip())
    return 0.5


def _read_label_map(model_dir: str) -> Dict[str, str]:
    """
    Preferred: <model_dir>/label_map.json (e.g. {"0":"Non-frequent","1":"Frequent"})
    """
    p = os.path.join(model_dir, "label_map.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensuring keys are strings
        return {str(k): str(v) for k, v in data.items()}
    return {"0": "Non-frequent", "1": "Frequent"}


def load_artifacts(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    threshold = _read_threshold(model_dir)
    label_map = _read_label_map(model_dir)

    device = _get_device()
    model.to(device)

    return model, tokenizer, threshold, label_map, device


@torch.no_grad()
def predict_review_with_artifacts(
    text: str,
    model,
    tokenizer,
    threshold: float,
    label_map: Dict[str, str],
    device: torch.device,
    max_length: int = 128,
    model_dir: Optional[str] = None,
) -> Dict[str, object]:
    model_text = clean_text(text)

    enc = tokenizer(
        model_text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    prob_class1 = float(probs[1].item())
    pred = 1 if prob_class1 >= threshold else 0
    return {
        "prediction": label_map.get(str(pred), str(pred)),
        "prediction_id": pred,
        "prob_class1_frequent": prob_class1,
        "threshold_used": threshold,
        "model_dir": model_dir,
    }


@torch.no_grad()
def predict_review(
    text: str,
    model_dir: str,
    max_length: int = 128,
) -> Dict[str, object]:
    model, tokenizer, threshold, label_map, device = load_artifacts(model_dir)
    return predict_review_with_artifacts(
        text=text,
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        label_map=label_map,
        device=device,
        max_length=max_length,
        model_dir=model_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Predict Frequent vs Non-Frequent from a single review text using a fine-tuned BERT model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join("outputs", "ml", "bert_label_overall", "best_model"),
        help="Path to the fine-tuned model directory (contains config.json, pytorch_model.bin, tokenizer files).",
    )
    parser.add_argument("--text", type=str, default=None, help="Review text to classify. If omitted, you'll be prompted.")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max_length (should match training).")
    args = parser.parse_args()

    text = args.text
    if not text:
        print("Paste a review and press Enter:\n")
        text = input("> ").strip()

    out = predict_review(
        text=text,
        model_dir=args.model_dir,
        max_length=args.max_length,
    )
    print("\nPrediction:", out["prediction"])
    print("P(Frequent=1):", round(out["prob_class1_frequent"], 6))
    print("Threshold used:", out["threshold_used"])
    print("Model dir:", out["model_dir"])


if __name__ == "__main__":
    main()
