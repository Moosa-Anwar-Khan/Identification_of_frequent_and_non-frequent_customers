import io
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request

from .config import (
    BERT_MAX_LENGTH,
    ML_DIR,
    PROJECT_ROOT,
    UI_MODEL_SUBDIR,
    UI_TEXT_INTENT_THRESHOLD,
    UI_UNCERTAIN_MARGIN,
)
from .predict_review import (
    load_artifacts,
    predict_review_with_artifacts,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "ui", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "ui", "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

MODEL_CHOICES = {
    "text_intent_k5_best_model": {
        "label": "Text-intent k5 best model",
        "subdir": os.path.join("bert_text_intent_k5_best_model", "best_model"),
        "threshold_override": UI_TEXT_INTENT_THRESHOLD,
    },
}
MODEL_CHOICE_ORDER = ["text_intent_k5_best_model"]
DEFAULT_MODEL_KEY = "text_intent_k5_best_model"

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, float], Dict[str, object]] = {}


def _latest_mtime(path: Path) -> float:
    latest = 0.0
    for item in path.rglob("*"):
        if item.is_file():
            ts = item.stat().st_mtime
            if ts > latest:
                latest = ts
    if latest == 0.0:
        latest = path.stat().st_mtime
    return latest


def _find_latest_model_dir() -> Optional[Tuple[Path, float]]:
    root = Path(ML_DIR)
    if not root.exists():
        return None

    if UI_MODEL_SUBDIR:
        preferred = root / UI_MODEL_SUBDIR
        if preferred.exists() and preferred.is_dir():
            return preferred, _latest_mtime(preferred)

    candidates = []
    for candidate in root.rglob("best_model"):
        if not candidate.is_dir():
            continue
        if not (candidate / "config.json").exists():
            continue
        candidates.append((candidate, _latest_mtime(candidate)))

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[1])


def _resolve_model_choice(model_key: Optional[str]) -> Tuple[Path, float, str]:
    key = model_key if model_key in MODEL_CHOICES else DEFAULT_MODEL_KEY
    choice = MODEL_CHOICES[key]

    if choice["subdir"]:
        model_dir = Path(ML_DIR) / choice["subdir"]
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        return model_dir, _latest_mtime(model_dir), choice["label"]

    latest = _find_latest_model_dir()
    if not latest:
        raise FileNotFoundError("No trained best_model directory found under outputs/ml.")
    model_dir, mtime = latest
    return model_dir, mtime, choice["label"]


def _load_artifacts_for_dir(model_dir: Path, mtime: float) -> Dict[str, object]:
    signature = (str(model_dir), mtime)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(signature)
        if cached:
            return cached

        model, tokenizer, threshold, label_map, device = load_artifacts(str(model_dir))
        state = {
            "signature": signature,
            "model": model,
            "tokenizer": tokenizer,
            "threshold": threshold,
            "label_map": label_map,
            "device": device,
        }
        _MODEL_CACHE[signature] = state
        return state


def _load_model(model_key: Optional[str]) -> Dict[str, object]:
    model_dir, mtime, model_label = _resolve_model_choice(model_key)
    artifacts = _load_artifacts_for_dir(model_dir, mtime)
    choice = MODEL_CHOICES.get(model_key or DEFAULT_MODEL_KEY, {})
    threshold_override = choice.get("threshold_override")
    if threshold_override is not None:
        artifacts = {**artifacts, "threshold": float(threshold_override)}
    display_path = os.path.relpath(model_dir, PROJECT_ROOT)
    model_updated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    return {
        **artifacts,
        "model_dir": str(model_dir),
        "model_dir_display": display_path,
        "model_label": model_label,
        "model_updated_at": model_updated_at,
    }


@app.get("/")
def index():
    model_choices = [
        {"key": key, "label": MODEL_CHOICES[key]["label"]}
        for key in MODEL_CHOICE_ORDER
        if key in MODEL_CHOICES
    ]
    return render_template(
        "index.html",
        model_choices=model_choices,
        default_model_key=DEFAULT_MODEL_KEY,
        default_model_label=MODEL_CHOICES[DEFAULT_MODEL_KEY]["label"],
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    model_key = payload.get("model_key") or DEFAULT_MODEL_KEY

    if not text:
        return jsonify({"error": "Please paste a review before running analysis."}), 400

    try:
        state = _load_model(model_key)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    result = predict_review_with_artifacts(
        text=text,
        model=state["model"],
        tokenizer=state["tokenizer"],
        threshold=state["threshold"],
        label_map=state["label_map"],
        device=state["device"],
        max_length=BERT_MAX_LENGTH,
        model_dir=state["model_dir_display"],
    )
    prob = float(result.get("prob_class1_frequent", 0.0))
    thr = float(result.get("threshold_used", 0.5))
    margin = float(UI_UNCERTAIN_MARGIN)
    if margin > 0 and thr > 0:
        margin = min(margin, max(0.01, thr * 0.5))
    is_uncertain = margin > 0 and (thr - margin) <= prob <= (thr + margin)
    if is_uncertain:
        result["prediction"] = "Non-frequent"
        result["prediction_id"] = 0
    result["is_uncertain"] = is_uncertain
    result["model_updated_at"] = state["model_updated_at"]
    result["model_dir"] = state["model_dir_display"]
    result["model_label"] = state["model_label"]

    return jsonify(result)


@app.post("/api/predict_csv")
def api_predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "Missing CSV file upload."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "Please select a CSV file to upload."}), 400

    try:
        df = pd.read_csv(file, header=None)
    except Exception:
        return jsonify({"error": "Unable to read CSV. Please upload a valid CSV file."}), 400

    if df.empty:
        return jsonify({"error": "Uploaded CSV is empty."}), 400

    texts = df.iloc[:, 0].fillna("").astype(str).tolist()
    if not any(t.strip() for t in texts):
        return jsonify({"error": "CSV contains no review text."}), 400

    try:
        state = _load_model(DEFAULT_MODEL_KEY)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    labels = []
    for text in texts:
        cleaned = text.strip()
        if not cleaned:
            labels.append("Non-Frequent")
            continue
        result = predict_review_with_artifacts(
            text=cleaned,
            model=state["model"],
            tokenizer=state["tokenizer"],
            threshold=state["threshold"],
            label_map=state["label_map"],
            device=state["device"],
            max_length=BERT_MAX_LENGTH,
            model_dir=state["model_dir_display"],
        )
        labels.append(result.get("prediction", "Non-Frequent"))

    out_df = pd.DataFrame({"review": texts, "label_intent": labels})
    buffer = io.StringIO()
    out_df.to_csv(buffer, index=False)
    buffer.seek(0)

    return app.response_class(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=labeled_reviews.csv"},
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug)
