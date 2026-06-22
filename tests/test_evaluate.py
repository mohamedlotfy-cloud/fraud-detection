"""
test_evaluate.py — Unit tests for evaluate.py

Tests evaluate_model() and print_example_cases() using a tiny synthetic
dataset and simple sklearn models. No real creditcard.csv needed.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import evaluate   # import the module so we can patch its namespace
from evaluate import evaluate_model, print_example_cases


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """200 samples, 30 features, 20 fraud rows."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 30))
    y = np.zeros(n, dtype=int)
    y[:20] = 1
    rng.shuffle(y)
    col_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    X_df = pd.DataFrame(X, columns=col_names)
    y_s  = pd.Series(y)
    return X_df, y_s


@pytest.fixture(scope="module")
def fitted_lr(synthetic_data):
    X, y = synthetic_data
    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_dummy(synthetic_data):
    X, y = synthetic_data
    model = DummyClassifier(strategy="stratified", random_state=0)
    model.fit(X, y)
    return model


# ── Tests: evaluate_model() ───────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_dict(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """evaluate_model must return a dict."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("logistic_regression", fitted_lr, X, y)
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """Result must contain Model, Precision, Recall, F1, ROC-AUC."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("logistic_regression", fitted_lr, X, y)
        for key in ["Model", "Precision", "Recall", "F1", "ROC-AUC"]:
            assert key in result, f"Missing key: {key}"

    def test_model_name_in_result(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """Model name in result must match the name argument."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("my_model", fitted_lr, X, y)
        assert result["Model"] == "my_model"

    def test_metrics_between_0_and_1(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """Precision, Recall, F1, ROC-AUC must all be in [0, 1]."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("logistic_regression", fitted_lr, X, y)
        for key in ["Precision", "Recall", "F1", "ROC-AUC"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    def test_confusion_matrix_png_created(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """evaluate_model must save a confusion matrix PNG."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        evaluate_model("test_model", fitted_lr, X, y)
        expected = os.path.join(str(tmp_path), "test_model_confusion_matrix.png")
        assert os.path.exists(expected), "Confusion matrix PNG not created"

    def test_dummy_classifier_works(self, fitted_dummy, synthetic_data, tmp_path, monkeypatch):
        """evaluate_model should work with any sklearn-compatible classifier."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("dummy", fitted_dummy, X, y)
        assert isinstance(result, dict)
        assert result["Model"] == "dummy"

    def test_f1_rounded_to_4_decimals(self, fitted_lr, synthetic_data, tmp_path, monkeypatch):
        """Metric values must be rounded to 4 decimal places."""
        X, y = synthetic_data
        monkeypatch.setattr(evaluate, "FIGURES_DIR", str(tmp_path))
        result = evaluate_model("logistic_regression", fitted_lr, X, y)
        for key in ["Precision", "Recall", "F1", "ROC-AUC"]:
            val = result[key]
            assert val == round(val, 4), f"{key} not rounded to 4 decimals"


# ── Tests: print_example_cases() ─────────────────────────────────────────────

class TestPrintExampleCases:

    def test_runs_without_error(self, fitted_lr, synthetic_data):
        """print_example_cases must not raise any exception."""
        X, y = synthetic_data
        print_example_cases("logistic_regression", fitted_lr, X, y)

    def test_output_contains_model_name(self, fitted_lr, synthetic_data, capsys):
        """Printed output must include the model name."""
        X, y = synthetic_data
        print_example_cases("logistic_regression", fitted_lr, X, y)
        captured = capsys.readouterr()
        assert "logistic_regression" in captured.out

    def test_output_contains_caught_or_missed(self, fitted_lr, synthetic_data, capsys):
        """Output must mention caught or missed fraud cases."""
        X, y = synthetic_data
        print_example_cases("logistic_regression", fitted_lr, X, y)
        captured = capsys.readouterr()
        assert "CAUGHT" in captured.out or "MISSED" in captured.out
