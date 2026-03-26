"""
features.py — Feature pipeline & cohort assignment for the Financial Historian
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import MiniBatchKMeans

from config import ALL_FEATURES, T0_FEATURES, TARGET, TRAIN_YEARS, SEED


def load_and_filter(path: str) -> pd.DataFrame:
    """Load preprocessed parquet/csv, apply post-2012 macro filter, fix types."""
    print("⏳ Loading data …")
    if str(path).endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Fix term_months if stored as string e.g. " 36 months"
    if 'term_months' in df.columns and df['term_months'].dtype == object:
        df['term_months'] = df['term_months'].str.extract(r'(\d+)').astype(float)

    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    df['issue_year'] = df['issue_d'].dt.year

    before = len(df)
    df = df[df['issue_year'] >= 2012].copy()
    print(f"✅ Loaded {before:,} rows → {len(df):,} after post-2012 macro filter")
    return df


def impute(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Median imputation on any remaining NaN in numeric feature columns."""
    for col in features:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        med = df[col].median()
        df[col] = df[col].replace([np.inf, -np.inf], med).fillna(med)
    return df


def temporal_split(df: pd.DataFrame):
    """Walk-forward temporal split — no future leaking into past folds."""
    train = df[df['issue_year'].isin(TRAIN_YEARS)]
    val   = df[df['issue_year'] == 2017]
    test  = df[df['issue_year'] == 2018]
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def scale_features(X_train, X_val, X_test):
    """RobustScaler — handles outliers better than StandardScaler."""
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def build_cohorts(X_train_scaled: np.ndarray, n_clusters: int = 20) -> MiniBatchKMeans:
    """
    Build cohort clusters using T=0 origination features only.
    Returns fitted KMeans model for cohort assignment.
    """
    print(f"⏳ Building {n_clusters} cohort clusters …")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=10000)
    kmeans.fit(X_train_scaled)
    print("✅ Cohorts built")
    return kmeans


def compute_cohort_risk(df_train: pd.DataFrame, cohort_labels: np.ndarray) -> dict:
    """
    For each cohort: compute historical delinquency rate.
    Returns dict {cohort_id: delinquency_rate}
    """
    df_tmp = df_train[[TARGET]].copy()
    df_tmp['cohort'] = cohort_labels
    cohort_risk = df_tmp.groupby('cohort')[TARGET].mean().to_dict()
    return cohort_risk


def apply_threshold_rules(df: pd.DataFrame) -> pd.Series:
    """
    Layer 2: Absolute threshold rules — cohort-independent danger signals.
    Returns a score 0–1 (proportion of thresholds breached).
    """
    rules = [
        df['fico_drop']      > 40,
        df['late_fee_flag']  == 1,
        df['revol_util']     > 70,
        df['inq_pressure']   > 0.15,
        df['dti']            > 22,
        df['repay_ratio']    < 0.30,
    ]
    # Only apply rules for columns that exist
    valid_rules = []
    for r in rules:
        try:
            valid_rules.append(r.astype(int))
        except Exception:
            pass
    if not valid_rules:
        return pd.Series(0, index=df.index)
    score = sum(valid_rules) / len(valid_rules)
    return score
