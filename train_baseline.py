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


# ...existing code...
def train(train_csv, out_path=None, n_splits=5, svd_components=200, test_csv='dataset/test.csv', test_out='test_out.csv'):
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

    df = df[~df['price'].isna()].copy()
    df['log_price'] = np.log1p(df['price'].astype(float))
    y = df['log_price'].values

    # TF-IDF (tighter control) + SVD
    print('Fitting TF-IDF...')
    tf = TfidfVectorizer(ngram_range=(1,3), analyzer='word', max_features=50000, min_df=5, stop_words='english')
    X_text = tf.fit_transform(df['content_clean'].fillna(''))

    print('Reducing TF-IDF with TruncatedSVD to', svd_components)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_text_reduced = svd.fit_transform(X_text)

    # tabular
    X_tab = df[['ipq','title_len','desc_len']].fillna(0).values

    # final features (dense)
    X = np.hstack([X_text_reduced, X_tab])
    print('Feature shape', X.shape)

    # stratify by binned target for folds
    y_bins = pd.qcut(df['log_price'], q=10, duplicates='drop', labels=False)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(df))
    fold_models = []
    fold = 0
    for train_idx, val_idx in skf.split(X, y_bins):
        fold += 1
        print(f'Fold {fold}')
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.03,
            'num_leaves': 64,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'seed': 42,
            'verbose': -1
        }
        callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)]
        model = lgb.train(params, lgb_train, valid_sets=[lgb_val], callbacks=callbacks, num_boost_round=3000)
        fold_models.append(model)

        preds_log = model.predict(X_val)
        oof_preds[val_idx] = preds_log

        preds = np.expm1(preds_log)
        y_true = np.expm1(y_val)
        mae = mean_absolute_error(y_true, preds)
        smape_score = smape(y_true, preds)
        print(f'Fold {fold} MAE: {mae:.4f} SMAPE: {smape_score:.4f}')

    # overall oof
    oof_preds_exp = np.expm1(oof_preds)
    y_true_all = np.expm1(y)
    print('CV MAE:', mean_absolute_error(y_true_all, oof_preds_exp))
    print('CV SMAPE:', smape(y_true_all, oof_preds_exp))

    # Instead of saving models to disk, use the CV ensemble to predict on test set
    # and write results to `test_out.csv` at project root.
    try:
        print('Loading test file for predictions:', test_csv)
        df_test = pd.read_csv(test_csv)
        # build test features using same tfidf+svd pipeline
        print('Preparing test features...')
        if 'catalog_content' in df_test.columns:
            df_test['catalog_content'] = df_test['catalog_content'].fillna('')
        else:
            # fallback: use first object column or empty strings
            obj_cols = [c for c in df_test.columns if df_test[c].dtype == 'object']
            if obj_cols:
                df_test['catalog_content'] = df_test[obj_cols[0]].fillna('')
            else:
                df_test['catalog_content'] = ''
        df_test['content_clean'] = df_test['catalog_content'].apply(clean_text)
        X_text_test = tf.transform(df_test['content_clean'].fillna(''))
        X_text_test_reduced = svd.transform(X_text_test)
        X_tab_test = df_test[['catalog_content']].copy()
        # create simple tabular features (ipq, title_len, desc_len)
        df_test['ipq'] = df_test['catalog_content'].apply(extract_ipq)
        df_test['title_len'] = df_test['content_clean'].apply(lambda x: len(x.split('\n')[0]) if x else 0)
        df_test['desc_len'] = df_test['content_clean'].apply(len)
        X_tab = df_test[['ipq','title_len','desc_len']].fillna(0).values
        X_test = np.hstack([X_text_test_reduced, X_tab])

        print('Predicting test set with CV ensemble...')
        preds_log_test = np.column_stack([m.predict(X_test) for m in fold_models]).mean(axis=1)
        preds_test = np.expm1(preds_log_test)

        # detect id column
        id_col = 'sample_id' if 'sample_id' in df_test.columns else df_test.columns[0]
        out_df = pd.DataFrame({'sample_id': df_test[id_col].values, 'price': preds_test.astype(float)})
        out_df.to_csv(test_out, index=False)
        print('Wrote test predictions to', test_out)
    except Exception as ex:
        print('Warning: could not generate test predictions:', ex)
# ...existing code...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='dataset/train.csv')
    parser.add_argument('--out', type=str, default='models/lgb_baseline.pkl')
    args = parser.parse_args()
    train(args.train, args.out)
