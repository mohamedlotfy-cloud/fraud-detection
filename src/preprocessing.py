"""
preprocessing.py — Load, scale, and split the dataset.

Public API:
    load_and_split(path) -> X_train, X_test, y_train, y_test
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import DATA_PATH, TEST_SIZE, RANDOM_STATE


def load_and_split(path=DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Load the raw CSV, apply feature scaling, and return a stratified train/test split.

    Why stratified?  With only ~0.17% fraud rows, a random split could accidentally
    put almost no fraud cases in the test set.  Stratified splitting guarantees that
    both halves keep the same fraud ratio as the full dataset.
    """

    # ── 1. Load ────────────────────────────────────────────────────────────
    print(f"Loading dataset from '{path}' ...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # ── 2. Separate features and target ───────────────────────────────────
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ── 3. Stratified train / test split ──────────────────────────────────
    # stratify=y ensures both splits have ~the same fraud percentage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 4. Scale Amount and Time ───────────────────────────────────────────
    # V1–V28 are already PCA-transformed and roughly standardised by the dataset
    # authors, but Amount and Time are raw and have very different magnitudes.
    # We fit the scaler ONLY on the training set to prevent data leakage.
    scaler = StandardScaler()
    for col in ["Amount", "Time"]:
        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col]  = scaler.transform(X_test[[col]])   # use train stats only

    # ── 5. Sanity-check fraud ratios ───────────────────────────────────────
    train_fraud_pct = 100 * y_train.sum() / len(y_train)
    test_fraud_pct  = 100 * y_test.sum()  / len(y_test)

    print(f"\n  Train set : {len(X_train):,} rows  |  fraud = {y_train.sum():,}  ({train_fraud_pct:.3f}%)")
    print(f"  Test  set : {len(X_test):,}  rows  |  fraud = {y_test.sum():,}   ({test_fraud_pct:.3f}%)")
    print(f"  Fraud ratio preserved — train={train_fraud_pct:.3f}%  test={test_fraud_pct:.3f}%  ✓")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_split()
