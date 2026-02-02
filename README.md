# Identification of Frequent and Non-Frequent  

**Team Project by:**
**Moosa Anwar Khan (moosa.anwar.khan@uni-potsdam.de) and Abdul Azeem Sikandar (sikandar@uni-potsdam.de)**

This repository contains the code, results, and figures for the research project: 

**"Identification of frequent and non-frequent customers"**

This project studies whether a text-only BERT classifier, trained on manually labeled Amazon All_Beauty reviews, can distinguish frequent vs non-frequent customer intent. It provides training code, evaluation outputs, and simple prediction interfaces (CLI and web UI). 

The project was conducted as part of the course “Applied AI and Data Science” at the University of Potsdam.

## Research question
Can a text-only BERT classifier, trained on manually labeled Amazon All_Beauty reviews, distinguish frequent/non-frequent customer intent?

## Summary
We fine-tune a BERT model on labeled review text and evaluate it with standard classification metrics. The repo also includes utilities to visualize results, run single or batch predictions, and serve a lightweight Flask UI for interactive use.

## Data and labels
- **Source**: Amazon All_Beauty reviews (stored under `data/` and `labeled_data/`).
- **Large raw files**: `All_Beauty.jsonl` and `meta_All_Beauty.jsonl` are excluded from this repo due to size. It can be downloaded from `https://amazon-reviews-2023.github.io/`.
- **Labels**: binary classes representing customer intent:
  - `1` = Frequent
  - `0` = Non-frequent
- The training pipeline expects a labeled CSV in `labeled_data/labeled_data.csv`.

## Model and approach
- **Model**: `bert-base-uncased` fine-tuned for binary text classification.
- **Inputs**: review text only (no metadata features).
- **Training**: stratified splits (with optional user-level splits to reduce leakage).
- **Evaluation**: precision/recall/F1, confusion matrix, and PR curve; threshold tuning for class 1.

## Outputs
Model artifacts and evaluation results are written to `outputs/`:
- `outputs/ml/`: saved models, thresholds, label maps
- `outputs/figures/`: confusion matrices and PR curves
- `outputs/tables/`: classification reports and threshold summary

## Project structure
- `src/bert_training.py`: trains the BERT classifier and exports metrics/plots.
- `src/pipeline_text_intent.py`: end-to-end training + export pipeline.
- `src/predict_review.py`: CLI prediction for a single review.
- `src/gui_app.py`: Flask UI for interactive predictions and CSV batch scoring.
- `src/config.py`: configuration for data paths, training hyperparameters, thresholds.

## Quick start
For install dependencies:
```bash
pip install -r requirements.txt
```

For training the text-intent model:
```bash
python -m src.pipeline_text_intent
```

For running the web UI:
```bash
python -m src.gui_app
```

## Note
The model is trained only "Amazon all beauty products" reviews. It might not perform well on reviews related to other product categories.

## Summary
We wanted to learn signals in review texts that indicate whether a customer sounds like a frequent buyer. The model is trained on labeled reviews, then tested and packaged so it can be used from the command line or a simple web page.
