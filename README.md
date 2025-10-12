# Smart Pricing — Multi-modal Price Prediction

This repository contains a baseline multi-modal solution for predicting product prices using product text (`catalog_content`), images, and engineered tabular features. The goal is to provide a clear, reproducible baseline and a scaffold to iterate toward stronger models (transformer-based text, CNN image models, ensembles).

## What this project contains

- `dataset/` — training and test CSVs you placed here (expected: `train.csv`, `test.csv`, `sample_test.csv`).
- `src/` — utility code shipped with the original project (e.g. `utils.py` for image downloads).
- `train_baseline.py` — lightweight baseline trainer that builds TF-IDF + simple engineered features and trains LightGBM on log-price. Saves a joblib model and TF-IDF vectorizer.
- `requirements.txt` — Python dependencies for reproducing the environment.
- `README.md` — this file (project overview, how to run)
- `.venv/` — (optional) virtual environment created during development (not required to be committed)


## Data description (expected columns)

The dataset should include at least the following columns in `dataset/train.csv`:

- `sample_id` — unique id for the sample
- `catalog_content` — combined title, description and other text fields
- `image_link` — public URL for product image (optional for baseline)
- `price` — target variable (only in training data)

The baseline training script only uses `catalog_content` (TF-IDF) and a few engineered numeric features extracted from the text (pack quantity, title/description length). The multi-modal branch (not yet fully implemented) would use `image_link` + a CNN and a transformer for text.

## Quickstart — Reproduce baseline (recommended)

1. Create and activate a Python virtual environment (macOS / zsh example):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2. Train baseline LightGBM model using the data in `dataset/train.csv`:

```bash
# run from repo root
python3 train_baseline.py --train dataset/train.csv --out models/lgb_baseline.pkl
```

The script will:
- Read `dataset/train.csv` and extract: `ipq` (pack quantity), `title_len`, `desc_len` from `catalog_content`.
- Fit a TF-IDF (1-2 grams) vectorizer on `catalog_content`.
- Train a LightGBM regressor on `log1p(price)` with early stopping.
- Save the trained model and vectorizer to the `models/` directory.

After training it prints validation metrics (MAE on original price scale and SMAPE).

## Files you can edit / extend

- `train_baseline.py` — modify preprocessing, add new engineered features or change LightGBM parameters.
- `src/utils.py` — contains an image downloader utility. Use this to download images when building a multi-modal dataset.
- Add `src/dataset.py` and `src/model.py` when you implement the PyTorch multi-modal pipeline (I can help scaffold these).

## Recommended next steps (how to improve)

1. Implement a PyTorch dataset that:
   - downloads and preprocesses images (resize, normalize),
   - tokenizes `catalog_content` with a transformer tokenizer (e.g. `all-MiniLM-L6-v2` or `deberta-v3-base`),
   - supplies engineered numeric/categorical features.
2. Implement a multi-modal model that fuses transformer text embeddings + CNN image features + tabular features (I provided a scaffold in earlier iterations).
3. Train LightGBM and the neural model and ensemble their predictions (simple weighted average or stacking).
4. Run hyperparameter tuning (Optuna) for LightGBM and neural network learning rate / projection sizes.

## Notes and troubleshooting

- On macOS, LightGBM may require `libomp`. If you see an error loading `lib_lightgbm.dylib`, install it with Homebrew:

```bash
brew install libomp
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

- The project intentionally keeps the baseline simple so you can iterate quickly. If you want, I can implement the multi-modal training pipeline next and run a short training pass on your dataset.

## Contact / Next action

If you want me to proceed I can:
- run the baseline training end-to-end and report final validation metrics (I will use `dataset/train.csv`), or
- implement the PyTorch multi-modal dataset and training loop and run a smoke test.

Tell me which of these you'd like done next and I'll proceed.
# ML Challenge 2025 Problem Statement

## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 75k products with complete product details and prices
- **Test Set:** 75k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.

Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

### File Descriptions:

*Source files*

1. **src/utils.py:** Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
2. **sample_code.py:** Sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
3. **dataset/sample_test.csv:** Sample test input file.
4. **dataset/sample_test_out.csv:** Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

### Constraints:

1. You will be provided with a sample output file. Format your output to match the sample output file exactly. 

2. Predicted prices must be positive float values.

3. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

### Evaluation Criteria:

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**: A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120  
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

### Leaderboard Information:

- **Public Leaderboard:** During the challenge, rankings will be based on 25K samples from the test set to provide real-time feedback on your model's performance.
- **Final Rankings:** The final decision will be based on performance on the complete 75K test set along with provided documentation of the proposed approach by the teams.

### Submission Requirements:

1. Upload a `test_out.csv` file in the Portal with the exact same formatting as `sample_test_out.csv`

2. All participating teams must also provide a 1-page document describing:
   - Methodology used
   - Model architecture/algorithms selected
   - Feature engineering techniques applied
   - Any other relevant information about the approach
   Note: A sample template for this documentation is provided in Documentation_template.md

### **Academic Integrity and Fair Play:**

**⚠️ STRICTLY PROHIBITED: External Price Lookup**

Participants are **STRICTLY NOT ALLOWED** to obtain prices from the internet, external databases, or any sources outside the provided dataset. This includes but is not limited to:
- Web scraping product prices from e-commerce websites
- Using APIs to fetch current market prices
- Manual price lookup from online sources
- Using any external pricing databases or services

**Enforcement:**
- All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
- Any evidence of external price lookup or data augmentation from internet sources will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the provided training data. External price lookup defeats the purpose of the challenge.


### Tips for Success:

- Consider both textual features (catalog_content) and visual features (product images)
- Explore feature engineering techniques for text and image data
- Consider ensemble methods combining different model types
- Pay attention to outliers and data preprocessing
