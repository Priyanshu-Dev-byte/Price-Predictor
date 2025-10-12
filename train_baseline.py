import os
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from scipy import sparse


def extract_ipq(text):
    if not isinstance(text, str):
        return 1
    patterns = [r'IPQ[:\s]*([0-9]+)', r'Pack of[:\s]*([0-9]+)', r'(\d+)\s*units?', r'(\d+)\s*ct', r'(\d+)\s*count']
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return 1


def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def smape(a, f):
    # a = actual, f = forecast
    a = np.array(a)
    f = np.array(f)
    denom = (np.abs(a) + np.abs(f))
    mask = denom == 0
    res = np.zeros_like(a, dtype=float)
    res[~mask] = np.abs(f[~mask] - a[~mask]) / denom[~mask]
    return 200.0 * np.mean(res)


def train(train_csv, out_path):
    print('Loading', train_csv)
    df = pd.read_csv(train_csv)
    if 'price' not in df.columns:
        raise ValueError('train csv must have a price column')

    # basic preprocessing
    df['catalog_content'] = df.get('catalog_content', '').fillna('')
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['content_clean'] = df['catalog_content'].apply(clean_text)
    df['title_len'] = df['content_clean'].apply(lambda x: len(x.split('\n')[0]) if x else 0)
    df['desc_len'] = df['content_clean'].apply(len)

    # target
    df = df[~df['price'].isna()].copy()
    df['log_price'] = np.log1p(df['price'].astype(float))

    # TF-IDF
    print('Fitting TF-IDF...')
    tf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    X_text = tf.fit_transform(df['content_clean'].fillna(''))

    # tabular
    X_tab = df[['ipq','title_len','desc_len']].fillna(0).values

    X = sparse.hstack([X_text, sparse.csr_matrix(X_tab)])
    y = df['log_price'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training LightGBM...')
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42,
        'verbose': -1
    }
    # use callback-based early stopping for recent lightgbm versions
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    model = lgb.train(params, lgb_train, valid_sets=[lgb_val], callbacks=callbacks, num_boost_round=2000)

    # predict on val
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_val)

    mae = mean_absolute_error(y_true, preds)
    smape_score = smape(y_true, preds)

    print(f'Validation MAE: {mae:.4f}')
    print(f'Validation SMAPE: {smape_score:.4f}')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump({'model': model, 'tfidf': tf}, out_path)
    print('Model saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='dataset/train.csv')
    parser.add_argument('--out', type=str, default='models/lgb_baseline.pkl')
    args = parser.parse_args()
    train(args.train, args.out)
