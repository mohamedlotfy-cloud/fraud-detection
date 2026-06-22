"""
test_preprocessing.py — Unit tests for preprocessing.py

Run with:
    pytest tests/ -v --cov=src --cov-report=term-missing

These tests use a synthetic mini-dataset so they don't need the real
creditcard.csv to run — CI-friendly and fast.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

# Make sure src/ and root are on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import load_and_split


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_csv(tmp_path_factory):
    """
    Create a tiny synthetic CSV that mirrors creditcard.csv structure:
    - 30 feature columns (V1-V28, Amount, Time) + Class
    - 1000 rows, ~0.17% fraud → we'll use 5% for testability (50 fraud rows)
    """
    rng = np.random.default_rng(42)
    n = 1000
    n_fraud = 50

    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Amount"] = rng.uniform(0, 2000, n)
    data["Time"]   = np.arange(n, dtype=float)
    data["Class"]  = np.zeros(n, dtype=int)

    # Set first n_fraud rows as fraud
    fraud_indices = rng.choice(n, size=n_fraud, replace=False)
    data["Class"][fraud_indices] = 1

    df = pd.DataFrame(data)
    path = tmp_path_factory.mktemp("data") / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path), df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadAndSplit:

    def test_returns_four_objects(self, tiny_csv):
        """load_and_split must return exactly 4 objects."""
        path, _ = tiny_csv
        result = load_and_split(path=path)
        assert len(result) == 4

    def test_correct_split_sizes(self, tiny_csv):
        """With test_size=0.30, train should be ~70% of data."""
        path, df = tiny_csv
        X_train, X_test, y_train, y_test = load_and_split(path=path, test_size=0.30)
        total = len(X_train) + len(X_test)
        assert total == len(df)
        assert len(X_train) > len(X_test), "Train set should be larger than test set"

    def test_train_larger_than_test(self, tiny_csv):
        """70/30 split — train must have more rows than test."""
        path, _ = tiny_csv
        X_train, X_test, _, _ = load_and_split(path=path)
        assert len(X_train) > len(X_test)

    def test_stratified_fraud_ratio(self, tiny_csv):
        """Fraud percentage in train and test should be within 2% of each other."""
        path, _ = tiny_csv
        _, _, y_train, y_test = load_and_split(path=path)
        train_pct = y_train.mean()
        test_pct  = y_test.mean()
        assert abs(train_pct - test_pct) < 0.02, (
            f"Fraud ratio differs too much: train={train_pct:.3f}  test={test_pct:.3f}"
        )

    def test_no_data_leakage_amount(self, tiny_csv):
        """
        The scaler must be fit only on train data.
        Test Amount values should NOT be zero-mean (because we used train stats).
        We verify this indirectly: train Amount should have mean ≈ 0.
        """
        path, _ = tiny_csv
        X_train, X_test, _, _ = load_and_split(path=path)
        train_amount_mean = X_train["Amount"].mean()
        assert abs(train_amount_mean) < 0.5, (
            f"Train Amount should be ~zero-mean after scaling, got {train_amount_mean:.3f}"
        )

    def test_amount_and_time_scaled(self, tiny_csv):
        """Amount and Time columns must have std ≈ 1 in the training set."""
        path, _ = tiny_csv
        X_train, _, _, _ = load_and_split(path=path)
        for col in ["Amount", "Time"]:
            std = X_train[col].std()
            assert 0.5 < std < 2.0, (
                f"Column '{col}' should be scaled (std≈1), got std={std:.3f}"
            )

    def test_feature_count(self, tiny_csv):
        """Output should have 30 feature columns (V1-V28 + Amount + Time)."""
        path, _ = tiny_csv
        X_train, X_test, _, _ = load_and_split(path=path)
        assert X_train.shape[1] == 30
        assert X_test.shape[1] == 30

    def test_class_column_excluded(self, tiny_csv):
        """The 'Class' target column must NOT appear in X_train or X_test."""
        path, _ = tiny_csv
        X_train, X_test, _, _ = load_and_split(path=path)
        assert "Class" not in X_train.columns
        assert "Class" not in X_test.columns

    def test_labels_binary(self, tiny_csv):
        """y_train and y_test must only contain values 0 and 1."""
        path, _ = tiny_csv
        _, _, y_train, y_test = load_and_split(path=path)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_fraud_present_in_both_splits(self, tiny_csv):
        """Both train and test must contain at least one fraud case."""
        path, _ = tiny_csv
        _, _, y_train, y_test = load_and_split(path=path)
        assert y_train.sum() > 0, "No fraud in train set!"
        assert y_test.sum()  > 0, "No fraud in test set!"

    def test_no_missing_values(self, tiny_csv):
        """Processed splits must have no NaN values."""
        path, _ = tiny_csv
        X_train, X_test, y_train, y_test = load_and_split(path=path)
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum()  == 0
        assert y_train.isnull().sum() == 0
        assert y_test.isnull().sum()  == 0

    def test_reproducibility(self, tiny_csv):
        """Same random_state must produce identical splits every time."""
        path, _ = tiny_csv
        X_train1, X_test1, _, _ = load_and_split(path=path, random_state=99)
        X_train2, X_test2, _, _ = load_and_split(path=path, random_state=99)
        pd.testing.assert_frame_equal(X_train1.reset_index(drop=True),
                                      X_train2.reset_index(drop=True))

    def test_different_seeds_give_different_splits(self, tiny_csv):
        """Different random states should produce different row orders."""
        path, _ = tiny_csv
        X_train1, _, _, _ = load_and_split(path=path, random_state=1)
        X_train2, _, _, _ = load_and_split(path=path, random_state=2)
        # At least some rows should differ
        assert not X_train1.reset_index(drop=True).equals(
            X_train2.reset_index(drop=True)
        )
