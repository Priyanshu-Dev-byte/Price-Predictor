# ML Challenge 2025 — One‑Page Methodology & Model Summary

Team Name: ChaosCoders
Team Members: Abhay Sagar, Priyanshu Rathour and Sakshi Chopra.
Submission Date: 2025-10-13

# Smart Pricing — Model Description

This document summarizes the method used for the Smart Pricing baseline and the decisions made during preprocessing, feature engineering, training, and evaluation. It is intended to accompany the code and deliverables in this repository.

## 1. Problem and Objective
Predict the expected retail price for products using multimodal product information (text fields and tabular metadata). The baseline focuses on text + tabular features to produce a robust, reproducible regression model and CSV predictions containing `product_id` and `expected_price`.

## 2. High-level Methodology
- Preprocess and clean textual fields (title, description, catalog_content), extract a small set of engineered tabular features.
- Convert long-form text into compact numerical representations using TF-IDF followed by Truncated SVD. Optionally augment with sentence-transformer embeddings when available.
- Train a LightGBM regression model on log-transformed target (log1p(price)) using K-fold cross-validation and then retrain on full data.
- Export model artifacts and produce a predictions CSV for test rows containing `product_id` and `expected_price` (inverse-transformed with expm1).

## 3. Preprocessing & Feature Engineering
- Missing and type handling: All text fields are coerced to strings and NaNs replaced with empty strings.
- Text cleaning: lowercase, remove non-alphanumeric characters, collapse whitespace.
- IPQ extraction: heuristic regex extraction (patterns like `IPQ: 6`, `Pack of 4`, `4 units`) to infer pack quantity (feature `ipq`). Defaults to 1 when not found.
- Length features: `title_len` and `desc_len` (character counts).
- Text vectorization: TF-IDF (1-2 grams, max_features=20k). To reduce dimensionality and memory, TruncatedSVD projects TF-IDF to 256 dims (or fewer if TF-IDF dims smaller).
- Optional embeddings: `sentence-transformers/all-MiniLM-L6-v2` can be used if installed — encoded in batches to avoid memory spikes.
- Tabular scaling: StandardScaler applied to [ipq, title_len, desc_len].

## 4. Model & Training Details
- Model: LightGBM (gradient boosting decision trees) regression.
- Target transform: y_train = log1p(price). Predictions are inverse-transformed with expm1.
- CV: 5-fold KFold (shuffle=True, seed=42) to estimate performance and stabilize training.
- Training hyperparameters (baseline):
  - objective: regression
  - metric: rmse
  - learning_rate: 0.05
  - num_leaves: 127
  - feature_fraction: 0.8
  - bagging_fraction: 0.8
  - bagging_freq: 5
  - seed: 42
- Early stopping: callbacks used during CV (stopping_rounds=50). Final num_boost_round chosen from mean best_iteration across CV folds (fallback 1000).

## 5. Evaluation
- During CV we compute: MAE (on original price scale after inverse transform) and SMAPE (symmetric mean absolute percentage error). These provide interpretable performance measures for pricing.
- The notebook and `train_baseline.py` print per-fold MAE/SMAPE and CV mean scores.

## 6. Artifacts produced
- `models/artifacts.joblib` — dict containing `final_model`, `tfidf`, `svd`, `scaler`, and metadata flags.
- `models/tfidf_vectorizer.joblib`, `models/svd.joblib`, `models/scaler.joblib` — saved separately for easier inference.
- `predictions.csv` — when a test/sample file is provided, contains `product_id` and `expected_price`.

## 7. How to run (local / Colab)
1. Upload `train.csv` to the working directory (or `dataset/train.csv`). Required columns: `price` (float) and preferably `product_id`. Optional: `title`, `description`, `catalog_content`.
2. Run the `run_local_train.ipynb` notebook (cells top-to-bottom) or run the trainer script directly:

```bash
# inside the repository root and with the environment activated
python3 train_baseline.py --train dataset/train.csv --out models/artifacts.joblib
```

3. If you have `sample_test.csv` or `test.csv` (with `product_id` and text fields), upload it and re-run the final notebook cell to generate `predictions.csv`.

Notes for macOS: LightGBM may require `libomp` to be installed (Homebrew: `brew install libomp`). On Linux/Colab the notebook installs `libomp-dev` automatically.

## 8. Assumptions & Limitations
- This baseline uses only textual and simple tabular features; it does not use image inputs. A full multimodal model (image + text) is expected to further improve accuracy.
- The IPQ heuristic is simple and may mis-extract in noisy descriptions.
- TF-IDF + SVD plus LightGBM is a strong baseline for tabular+text but deeper fine-tuning (e.g., transformer fine-tuning or ensembling with CatBoost/XGBoost/NN models) is likely to improve results.

## 9. Next Steps (recommended)
- Add image features: precompute CNN embeddings (ResNet/EfficientNet) and fuse with text features in a neural or gradient-boosting model.
- Run Optuna hyperparameter search over LightGBM params (learning_rate, num_leaves, max_depth, feature_fraction, bagging_fraction) with time-limited trials.
- Try stacking an ensemble of LightGBM + CatBoost + a small neural net on the concatenated features.

For any questions or to request a reproducible evaluation run with hyperparameter tuning, artifacts upload, or a multimodal implementation, contact the authors via the repository issues or provide instructions and we will extend the code accordingly.

## Drive Link 
https://drive.google.com/drive/folders/1PGFpz8N49vzHv04R7ofcNaYLSWhiAszP?usp=sharing
