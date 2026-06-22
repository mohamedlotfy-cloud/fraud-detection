"""
test_models.py — Unit tests for model training and evaluation logic.

Tests that models can be instantiated, trained, and produce valid predictions
on a tiny synthetic dataset — no real creditcard.csv required.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import build_models


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_dataset():
    """
    200-sample dataset with 30 features (matching creditcard structure).
    20 fraud (class=1) and 180 normal (class=0).
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    y = np.zeros(200, dtype=int)
    y[:20] = 1          # 10% fraud — easier to detect than the real 0.17%
    rng.shuffle(y)
    return X, y


@pytest.fixture(scope="module")
def trained_models(tiny_dataset):
    """Train all models on the tiny dataset and return them."""
    X, y = tiny_dataset
    models = build_models()
    for name, model in models.items():
        model.fit(X, y)
    return models


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildModels:

    def test_returns_five_models(self):
        """build_models() must return exactly 5 model instances."""
        models = build_models()
        assert len(models) == 5

    def test_expected_keys_present(self):
        expected = {"dummy", "logistic_regression", "random_forest", "xgboost", "neural_network"}
        assert set(build_models().keys()) == expected

    def test_all_have_fit_method(self):
        for name, model in build_models().items():
            assert hasattr(model, "fit"), f"'{name}' missing .fit()"

    def test_all_have_predict_method(self):
        for name, model in build_models().items():
            assert hasattr(model, "predict"), f"'{name}' missing .predict()"

    def test_all_have_predict_proba_method(self):
        for name, model in build_models().items():
            assert hasattr(model, "predict_proba"), f"'{name}' missing .predict_proba()"


class TestModelPredictions:

    def test_predict_returns_correct_shape(self, trained_models, tiny_dataset):
        """predict() must return one label per sample."""
        X, y = tiny_dataset
        for name, model in trained_models.items():
            preds = model.predict(X)
            assert preds.shape == (len(X),), f"'{name}' predict shape mismatch"

    def test_predict_only_binary_labels(self, trained_models, tiny_dataset):
        """All predictions must be 0 or 1."""
        X, y = tiny_dataset
        for name, model in trained_models.items():
            preds = model.predict(X)
            unique = set(preds)
            assert unique.issubset({0, 1}), f"'{name}' produced non-binary labels: {unique}"

    def test_predict_proba_shape(self, trained_models, tiny_dataset):
        """predict_proba() must return (n_samples, 2) for binary classification."""
        X, y = tiny_dataset
        for name, model in trained_models.items():
            proba = model.predict_proba(X)
            assert proba.shape == (len(X), 2), f"'{name}' predict_proba shape mismatch"

    def test_predict_proba_sums_to_one(self, trained_models, tiny_dataset):
        """Each row of predict_proba must sum to 1.0 (within floating point tolerance)."""
        X, _ = tiny_dataset
        for name, model in trained_models.items():
            proba = model.predict_proba(X)
            row_sums = proba.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), (
                f"'{name}' probabilities don't sum to 1"
            )

    def test_proba_between_0_and_1(self, trained_models, tiny_dataset):
        """All probability values must be in [0, 1]."""
        X, _ = tiny_dataset
        for name, model in trained_models.items():
            proba = model.predict_proba(X)
            assert proba.min() >= 0.0 and proba.max() <= 1.0, (
                f"'{name}' has probabilities outside [0, 1]"
            )

    def test_non_dummy_models_detect_some_fraud(self, trained_models, tiny_dataset):
        """
        Real models (not dummy) should detect at least some fraud.
        With 20% fraud in the tiny dataset this is a sanity check only.
        """
        X, y = tiny_dataset
        for name, model in trained_models.items():
            if name == "dummy":
                continue
            preds = model.predict(X)
            fraud_detected = ((preds == 1) & (y == 1)).sum()
            assert fraud_detected > 0, f"'{name}' detected zero fraud cases"


class TestModelPersistence:

    def test_model_can_be_saved_and_loaded(self, trained_models, tiny_dataset, tmp_path):
        """Models saved with joblib must load and produce identical predictions."""
        X, _ = tiny_dataset
        for name, model in trained_models.items():
            path = tmp_path / f"{name}.joblib"
            joblib.dump(model, str(path))
            assert path.exists(), f"Saved model file not found for '{name}'"

            loaded = joblib.load(str(path))
            original_preds = model.predict(X)
            loaded_preds   = loaded.predict(X)
            assert np.array_equal(original_preds, loaded_preds), (
                f"Loaded '{name}' predictions differ from original"
            )
