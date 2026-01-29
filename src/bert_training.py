import os
import json
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm

from .config import (
    ML_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE,
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_BATCH_SIZE,
    BERT_EPOCHS, BERT_LEARNING_RATE, BERT_WEIGHT_DECAY,
    BERT_WARMUP_RATIO, BERT_GRAD_CLIP_NORM, BERT_MAX_TRAIN_SAMPLES,
    BERT_FREEZE_BASE_MODEL, BERT_NUM_WORKERS, BERT_MAX_VAL_SAMPLES, BERT_MAX_TEST_SAMPLES,
    UNDERSAMPLE_NEG_PER_POS, UNDERSAMPLE_GROUP_COL_CANDIDATES,
    POSITIVE_CLASS_WEIGHT_MULTIPLIER, THRESHOLD_TUNING_BETA, THRESHOLD_TUNING_MIN_RECALL,
    USE_USER_LEVEL_SPLIT, CPU_NUM_THREADS, CPU_NUM_INTEROP_THREADS, TOKENIZERS_PARALLELISM,
)
from .ml_utils import (
    encode_binary_labels,
    best_threshold_by_fbeta,
    save_classification_reports,
    plot_confusion_matrix,
    plot_pr_curve,
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


def _get_device() -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


_THREADS_CONFIGURED = False


def _configure_threads():
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return
    if CPU_NUM_THREADS:
        torch.set_num_threads(int(CPU_NUM_THREADS))
    if CPU_NUM_INTEROP_THREADS:
        try:
            torch.set_num_interop_threads(int(CPU_NUM_INTEROP_THREADS))
        except RuntimeError as exc:
            print(f"[BERT] Warning: could not set interop threads: {exc}")
    if TOKENIZERS_PARALLELISM is not None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", str(TOKENIZERS_PARALLELISM))
    _THREADS_CONFIGURED = True


@torch.no_grad()
def _predict_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


@torch.no_grad()
def _eval_loss_and_proba(model, loader, loss_fn, device) -> Tuple[float, np.ndarray]:
    model.eval()
    losses = []
    probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            losses.append(float(loss_fn(logits, labels).item()))
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.detach().cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, (np.concatenate(probs) if probs else np.array([]))


def _split_stratified(
    X: pd.Series,
    y: np.ndarray,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _split_user_level(
    X: pd.Series,
    y: np.ndarray,
    groups: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Split by user_id to prevent leakage across a user's reviews."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test


def _undersample_majority_train(
    X_train: pd.Series,
    y_train: np.ndarray,
    group_train: Optional[pd.Series],
    neg_per_pos: int,
    seed: int,
) -> Tuple[pd.Series, np.ndarray, Optional[pd.Series]]:
    """Train-only majority-class undersampling (keeps all positives)."""
    rng = np.random.RandomState(seed)

    pos_mask = (y_train == 1)
    neg_mask = (y_train == 0)

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return X_train, y_train, group_train

    target_neg = min(n_neg, int(neg_per_pos) * n_pos)
    neg_indices = np.where(neg_mask)[0]

    if group_train is None:
        chosen_neg = rng.choice(neg_indices, size=target_neg, replace=False)
    else:
        g = group_train.fillna("__NA__").astype(str).to_numpy()
        neg_groups = g[neg_indices]
        # Allocate samples proportional to neg group sizes
        uniq, counts = np.unique(neg_groups, return_counts=True)
        proportions = counts / counts.sum()
        alloc = np.floor(proportions * target_neg).astype(int)
        # Fix rounding to hit target exactly
        while alloc.sum() < target_neg:
            alloc[np.argmax(proportions - (alloc / max(target_neg, 1)))] += 1
        while alloc.sum() > target_neg:
            alloc[np.argmax(alloc)] -= 1

        chosen_list = []
        for u, k in zip(uniq, alloc):
            idx_u = neg_indices[neg_groups == u]
            k = min(int(k), len(idx_u))
            if k > 0:
                chosen_list.append(rng.choice(idx_u, size=k, replace=False))
        chosen_neg = np.concatenate(chosen_list) if chosen_list else rng.choice(neg_indices, size=target_neg, replace=False)

        # If due to rounding/caps we got fewer than target_neg, top up randomly
        if len(chosen_neg) < target_neg:
            remaining = np.setdiff1d(neg_indices, chosen_neg, assume_unique=False)
            topup = rng.choice(remaining, size=(target_neg - len(chosen_neg)), replace=False)
            chosen_neg = np.concatenate([chosen_neg, topup])

    pos_indices = np.where(pos_mask)[0]
    keep_idx = np.concatenate([pos_indices, chosen_neg])
    rng.shuffle(keep_idx)

    X_train_u = X_train.iloc[keep_idx]
    y_train_u = y_train[keep_idx]
    group_u = group_train.iloc[keep_idx] if group_train is not None else None
    return X_train_u, y_train_u, group_u


def _cap_keep_positives(
    X: pd.Series,
    y: np.ndarray,
    group: Optional[pd.Series],
    max_samples: Optional[int],
    seed: int,
) -> Tuple[pd.Series, np.ndarray, Optional[pd.Series]]:
    """
    When a max sample cap is set, keep ALL positives and fill the remainder with negatives.
    This prevents rare positives from being dropped by random subsampling.
    """
    if max_samples is None or len(X) <= int(max_samples):
        return X, y, group

    rng = np.random.RandomState(seed)
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]

    n_pos = len(pos_idx)
    target_total = int(max_samples)

    if n_pos == 0:
        # No positives to preserve; fall back to random cap.
        idx = rng.choice(len(X), size=target_total, replace=False)
    else:
        keep_pos = pos_idx
        remaining = target_total - n_pos
        if remaining < 0:
            # More positives than the cap; downsample positives only.
            keep_pos = rng.choice(pos_idx, size=target_total, replace=False)
            remaining = 0

        chosen_neg = np.array([], dtype=int)
        if remaining > 0 and len(neg_idx) > 0:
            k = min(remaining, len(neg_idx))
            if group is None:
                chosen_neg = rng.choice(neg_idx, size=k, replace=False)
            else:
                g = group.fillna("__NA__").astype(str).to_numpy()
                neg_groups = g[neg_idx]
                uniq, counts = np.unique(neg_groups, return_counts=True)
                proportions = counts / counts.sum()
                alloc = np.floor(proportions * k).astype(int)
                # Fix rounding to hit target exactly
                while alloc.sum() < k:
                    alloc[np.argmax(proportions - (alloc / max(k, 1)))] += 1
                while alloc.sum() > k:
                    alloc[np.argmax(alloc)] -= 1

                chosen_list = []
                for u, k_u in zip(uniq, alloc):
                    idx_u = neg_idx[neg_groups == u]
                    k_u = min(int(k_u), len(idx_u))
                    if k_u > 0:
                        chosen_list.append(rng.choice(idx_u, size=k_u, replace=False))
                chosen_neg = (
                    np.concatenate(chosen_list)
                    if chosen_list
                    else rng.choice(neg_idx, size=k, replace=False)
                )

                # Top up if rounding produced fewer than k
                if len(chosen_neg) < k:
                    remaining_pool = np.setdiff1d(neg_idx, chosen_neg, assume_unique=False)
                    if len(remaining_pool) > 0:
                        topup = rng.choice(
                            remaining_pool,
                            size=(k - len(chosen_neg)),
                            replace=False,
                        )
                        chosen_neg = np.concatenate([chosen_neg, topup])

        idx = np.concatenate([keep_pos, chosen_neg]) if len(chosen_neg) > 0 else keep_pos

    rng.shuffle(idx)
    X_cap = X.iloc[idx]
    y_cap = y[idx]
    group_cap = group.iloc[idx] if group is not None else None
    return X_cap, y_cap, group_cap


def run_bert_training(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label_overall",
    out_subdir: Optional[str] = None,
):
    """
    BERT fine-tuning for binary classification with:
      - Stratified train/val/test split
      - Class-weighted loss
      - Train-only majority-class undersampling (NO oversampling)
      - Threshold tuning via PR curve / F1(class=1) on validation
      - Classification report and plots saved to outputs/ml

    Notes:
      - Validation and test loaders are NOT resampled.
      - Threshold is tuned on validation, then applied to test.
    """
    if text_col not in df.columns or label_col not in df.columns:
        print(
            f"Skipping BERT training for {label_col}: missing columns "
            f"(text_col={text_col!r} in df={text_col in df.columns}, "
            f"label_col={label_col!r} in df={label_col in df.columns})."
        )
        return

    _configure_threads()

    # Keep optional grouping columns for diversity-preserving undersampling
    extra_cols = [c for c in UNDERSAMPLE_GROUP_COL_CANDIDATES if c in df.columns]
    user_group_col = "user_id" if "user_id" in df.columns else None
    cols = [text_col, label_col, *extra_cols]
    if user_group_col and user_group_col not in cols:
        cols.insert(1, user_group_col)
    tmp = df[cols].dropna(subset=[text_col, label_col]).copy()
    if tmp.empty:
        print(f"Skipping BERT training for {label_col}: no rows after dropna on {text_col!r}/{label_col!r}.")
        return

    y, id2label = encode_binary_labels(tmp[label_col])
    X = tmp[text_col].astype(str)
    group_series = None
    for c in UNDERSAMPLE_GROUP_COL_CANDIDATES:
        if c in tmp.columns:
            group_series = tmp[c]
            break
    user_groups = tmp[user_group_col] if user_group_col and user_group_col in tmp.columns else None

    if USE_USER_LEVEL_SPLIT and user_groups is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = _split_user_level(X, y, user_groups)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = _split_stratified(X, y)
    group_train = group_series.loc[X_train.index] if group_series is not None else None

    # Optional subsampling for faster experimentation (random, stratified already)
    X_train, y_train, group_train = _cap_keep_positives(
        X_train,
        y_train,
        group_train,
        max_samples=BERT_MAX_TRAIN_SAMPLES,
        seed=RANDOM_STATE,
    )

    X_val, y_val, _ = _cap_keep_positives(
        X_val,
        y_val,
        None,
        max_samples=BERT_MAX_VAL_SAMPLES,
        seed=RANDOM_STATE,
    )

    X_test, y_test, _ = _cap_keep_positives(
        X_test,
        y_test,
        None,
        max_samples=BERT_MAX_TEST_SAMPLES,
        seed=RANDOM_STATE,
    )

    # Train-only majority-class undersampling (NO oversampling)
    X_train, y_train, group_train = _undersample_majority_train(
        X_train,
        y_train,
        group_train,
        neg_per_pos=int(UNDERSAMPLE_NEG_PER_POS),
        seed=RANDOM_STATE,
    )

    prev_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=2,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
    finally:
        hf_logging.set_verbosity(prev_verbosity)

    device = _get_device()
    print(f"[BERT] Using device: {device}")
    model.to(device)

    if BERT_FREEZE_BASE_MODEL:
        base_prefix = getattr(model, "base_model_prefix", None)
        base_model = getattr(model, base_prefix, None) if base_prefix else None
        if base_model is not None:
            for p in base_model.parameters():
                p.requires_grad = False

    # Class weights for loss (inverse frequency)
    class_counts = np.bincount(y_train, minlength=2).astype(float)
    class_weights = (class_counts.sum() / np.clip(class_counts, 1.0, None))
    class_weights = class_weights / class_weights.mean()  # normalize
    if POSITIVE_CLASS_WEIGHT_MULTIPLIER and POSITIVE_CLASS_WEIGHT_MULTIPLIER != 1.0:
        class_weights[1] *= float(POSITIVE_CLASS_WEIGHT_MULTIPLIER)
        class_weights = class_weights / class_weights.mean()
    # NOTE: We do NOT apply any oversampling multiplier; imbalance is handled via
    # train-only undersampling + (optional) class-weighted loss.
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # NOTE: We do NOT use WeightedRandomSampler here because it resamples with replacement
    # (i.e., it is a form of oversampling). We rely on undersampling instead.

    train_ds = TextDataset(X_train, y_train, tokenizer, max_length=BERT_MAX_LENGTH)
    val_ds = TextDataset(X_val, y_val, tokenizer, max_length=BERT_MAX_LENGTH)
    test_ds = TextDataset(X_test, y_test, tokenizer, max_length=BERT_MAX_LENGTH)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        train_ds,
        batch_size=BERT_BATCH_SIZE,
        shuffle=True,
        num_workers=BERT_NUM_WORKERS,
        collate_fn=data_collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=BERT_NUM_WORKERS,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=BERT_NUM_WORKERS,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LEARNING_RATE, weight_decay=BERT_WEIGHT_DECAY)

    total_steps = max(1, len(train_loader) * int(BERT_EPOCHS))
    warmup_steps = int(BERT_WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    out_dir = os.path.join(ML_DIR, out_subdir or f"bert_{label_col}")
    os.makedirs(out_dir, exist_ok=True)
    best_model_dir = os.path.join(out_dir, "best_model")

    best_val_loss = float("inf")
    best_val_fbeta = -1.0
    best_thr_for_best_fbeta = 0.5

    for epoch in range(int(BERT_EPOCHS)):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"[BERT] Train {label_col} epoch {epoch+1}", leave=False), start=1):
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=BERT_GRAD_CLIP_NORM)

            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            if step % 200 == 0:
                tqdm.write(f"[BERT] {label_col} epoch {epoch+1} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        avg_val_loss, val_prob = _eval_loss_and_proba(model, val_loader, loss_fn=loss_fn, device=device)
        thr, fbeta = best_threshold_by_fbeta(
            y_val,
            val_prob,
            beta=THRESHOLD_TUNING_BETA,
            min_recall=THRESHOLD_TUNING_MIN_RECALL,
        )

        print(
            f"[BERT] {label_col} epoch {epoch+1}/{int(BERT_EPOCHS)} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"best_thr_val={thr:.3f} best_fbeta_val={fbeta:.4f}"
        )

        with open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch+1} train_loss={avg_train_loss:.6f} "
                f"val_loss={avg_val_loss:.6f} best_thr_val={thr:.6f} best_fbeta_val={fbeta:.6f}\n"
            )

        if fbeta > best_val_fbeta:
            best_val_fbeta = float(fbeta)
            best_thr_for_best_fbeta = float(thr)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

    # Load best checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    # Predict probabilities
    val_prob = _predict_proba(model, val_loader, device=device)
    test_prob = _predict_proba(model, test_loader, device=device)

    best_thr_model, best_fbeta_model = best_threshold_by_fbeta(
        y_val,
        val_prob,
        beta=THRESHOLD_TUNING_BETA,
        min_recall=THRESHOLD_TUNING_MIN_RECALL,
    )

    best_thr = best_thr_model
    best_fbeta = best_fbeta_model

    # Default threshold 0.5
    y_pred_05 = (test_prob >= 0.5).astype(int)
    save_classification_reports(y_test, y_pred_05, out_dir, "bert_threshold_0p5", id2label=id2label)
    plot_confusion_matrix(
        y_test, y_pred_05,
        os.path.join(out_dir, "bert_confusion_matrix_threshold_0p5.png"),
        title=f"Confusion matrix ({label_col}) - BERT @0.5",
    )

    # Tuned threshold
    y_pred_best = (test_prob >= best_thr).astype(int)
    save_classification_reports(y_test, y_pred_best, out_dir, "bert_threshold_best", id2label=id2label)
    plot_confusion_matrix(
        y_test, y_pred_best,
        os.path.join(out_dir, "bert_confusion_matrix_threshold_best.png"),
        title=f"Confusion matrix ({label_col}) - BERT @best",
    )

    # Model-only reports for transparency
    y_pred_model_05 = (test_prob >= 0.5).astype(int)
    save_classification_reports(y_test, y_pred_model_05, out_dir, "bert_threshold_0p5_model_only", id2label=id2label)
    y_pred_model_best = (test_prob >= best_thr_model).astype(int)
    save_classification_reports(y_test, y_pred_model_best, out_dir, "bert_threshold_best_model_only", id2label=id2label)

    plot_pr_curve(
        y_test,
        test_prob,
        os.path.join(out_dir, "bert_precision_recall_curve.png"),
        title=f"Precision-Recall ({label_col}) - BERT",
        best_threshold=best_thr,
    )

    with open(os.path.join(out_dir, "bert_best_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_threshold={best_thr}\n")
        f.write(f"best_fbeta_on_val={best_fbeta}\n")
        f.write(f"model_only_best_threshold={best_thr_model}\n")
        f.write(f"model_only_best_fbeta_on_val={best_fbeta_model}\n")
        f.write(f"threshold_tuning_beta={THRESHOLD_TUNING_BETA}\n")
        f.write(f"threshold_tuning_min_recall={THRESHOLD_TUNING_MIN_RECALL}\n")
        f.write(f"class_counts_train={class_counts.tolist()}\n")
        f.write(f"class_weights_loss={class_weights.tolist()}\n")

    
    # Save inference artifacts inside the best_model directory (used by predict_review.py)
    try:
        os.makedirs(best_model_dir, exist_ok=True)
        with open(os.path.join(best_model_dir, "best_threshold.txt"), "w", encoding="utf-8") as f_thr:
            f_thr.write(str(best_thr))
        with open(os.path.join(best_model_dir, "label_map.json"), "w", encoding="utf-8") as f_map:
            json.dump({str(k): v for k, v in id2label.items()}, f_map, indent=2)
    except Exception:
        pass

    print(f"Saved BERT outputs to: {out_dir}")
