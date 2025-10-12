# Smart Pricing - Documentation

This repository contains a multi-modal solution for predicting product prices using textual (catalog_content), image, and engineered tabular features. The goal is to build a robust model that accurately suggests optimal prices.

## Contents
- `requirements.txt` - Python dependencies
- `src/data_utils.py` - helpers to read data and download images
- `src/preprocess.py` - text cleaning and IPQ extraction
- `src/features.py` - vectorizers and encoders
- `src/models.py` - PyTorch multi-modal model scaffold
- `src/train.py` - baseline LightGBM training script


## Quickstart
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train baseline LightGBM model on your `train.csv`:

```bash
python -m src.train --train_csv train.csv --out models/lgb_baseline.pkl
```

## Project Plan
1. Start with the LightGBM baseline using engineered tabular + TF-IDF features.
2. Add multi-modal PyTorch model using a lightweight sentence-transformer and ResNet.
3. Ensemble LightGBM and neural model predictions.
4. Hyperparameter tuning with Optuna and advanced augmentations.

## Evaluation
- Use log1p transformation on price during training.
- Evaluate using MAE on original price scale and SMAPE.

## Next steps
- Implement full PyTorch training loop, dataset class, and image downloader.
- Add inference and submission formatting script.
- Run Optuna for hyperparameter search and weight the ensemble on validation.

