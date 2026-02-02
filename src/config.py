import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR  = os.path.join(OUTPUT_DIR, "tables")
ML_DIR      = os.path.join(OUTPUT_DIR, "ml")

for d in [FIGURES_DIR, TABLES_DIR, ML_DIR]:
    os.makedirs(d, exist_ok=True)

# Performance / threading
CPU_NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", os.cpu_count() or 8))
CPU_NUM_INTEROP_THREADS = int(os.environ.get("CPU_NUM_INTEROP_THREADS", max(1, CPU_NUM_THREADS // 2)))
TOKENIZERS_PARALLELISM = os.environ.get("TOKENIZERS_PARALLELISM", "true")

REVIEWS_PATH = os.path.join(DATA_DIR, "All_Beauty.jsonl")
META_PATH    = os.path.join(DATA_DIR, "meta_All_Beauty.jsonl")

# Undersampling (train-only) for BERT on CPU

# Keeping ALL positives in the training split, and randomly sample negatives to:
#   negatives <= UNDERSAMPLE_NEG_PER_POS * positives
UNDERSAMPLE_NEG_PER_POS = 10

# When undersampling negatives, trying to preserve diversity by sampling within one of these groups (first match wins)
UNDERSAMPLE_GROUP_COL_CANDIDATES = ["main_category", "store", "parent_asin"]

RANDOM_STATE = 42

# ############################
# BERT training config
# ############################
# Data split proportions (stratified)
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # fraction of the remaining train set
# Using user-level split to avoid leakage across the same user's reviews
USE_USER_LEVEL_SPLIT = True

# Threshold tuning for class 1 (Frequent)
# beta > 1 favors recall and beta < 1 favors precision
THRESHOLD_TUNING_BETA = 3.0
# Optional minimum recall constraint when choosing the threshold (to ignore it we can set it None)
THRESHOLD_TUNING_MIN_RECALL = 0.80

# BERT (CPU-friendly defaults)
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128

BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1 
BERT_GRAD_CLIP_NORM = 1.0

# Increasing this will result in emphasizing Frequent class during training (1.0 = no extra boost)
POSITIVE_CLASS_WEIGHT_MULTIPLIER = 1.5

# DataLoader workers for tokenization/batching
BERT_NUM_WORKERS = int(os.environ.get("BERT_NUM_WORKERS", max(1, min(8, CPU_NUM_THREADS // 2))))

# Optional speed-up on CPU
# Freezing is much faster on CPU unfreeze it for best quality time/GPU (If GPU exists).
BERT_FREEZE_BASE_MODEL = False

# Training on a subset (recommended without GPU)
# Caps keep all positives and add negatives up to the limit (see _cap_keep_positives). "Cap" is max no. of samples in the split.
BERT_MAX_TRAIN_SAMPLES = 20000
BERT_MAX_VAL_SAMPLES = 5000
BERT_MAX_TEST_SAMPLES = 5000

# Default model used by the GUI (relative to outputs/ml)
UI_MODEL_SUBDIR = os.path.join("bert_text_intent_k5_best_model", "best_model")

# UI prediction tuning
# Marking uncertain predictions when probability is close to the decision threshold.
UI_UNCERTAIN_MARGIN = float(os.environ.get("UI_UNCERTAIN_MARGIN", "0.1"))
# Optional override for the text-intent model threshold in the UI.
UI_TEXT_INTENT_THRESHOLD = None
