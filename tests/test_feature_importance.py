"""
test_feature_importance.py — Unit tests for feature_importance.py

Tests plot_feature_importance() with a tiny synthetic Random Forest
saved to a temp directory. No real creditcard.csv or model needed.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import joblib
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.ensemble import RandomForestClassifier
import feature_importance   # import the module so we can patch its namespace
from feature_importance import plot_feature_importance


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_rf_and_data(tmp_path):
    """
    Train a tiny RF on synthetic data, save it to tmp_path/models/,
    and return the tmp paths for models and figures.
    """
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 30))
    y = np.zeros(n, dtype=int)
    y[:20] = 1
    rng.shuffle(y)

    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    # Save as both tuned and plain RF so tests can try both
    joblib.dump(rf, str(models_dir / "random_forest_tuned.joblib"))
    joblib.dump(rf, str(models_dir / "random_forest.joblib"))

    # Build a synthetic DataFrame that load_and_split would return
    col_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    X_df = pd.DataFrame(X, columns=col_names)
    y_s  = pd.Series(y)

    return {
        "rf": rf,
        "X_train": X_df,
        "y_train": y_s,
        "models_dir": str(models_dir),
        "figures_dir": str(figures_dir),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPlotFeatureImportance:

    def test_creates_feature_importance_png(self, synthetic_rf_and_data, monkeypatch):
        """plot_feature_importance must save feature_importance.png."""
        d = synthetic_rf_and_data
        monkeypatch.setattr(feature_importance, "MODELS_DIR",  d["models_dir"])
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", d["figures_dir"])
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 10)
        monkeypatch.setattr(
            feature_importance, "load_and_split",
            lambda: (d["X_train"], d["X_train"], d["y_train"], d["y_train"])
        )
        plot_feature_importance()
        assert os.path.exists(os.path.join(d["figures_dir"], "feature_importance.png"))

    def test_png_is_non_empty(self, synthetic_rf_and_data, monkeypatch):
        """feature_importance.png must be a real image."""
        d = synthetic_rf_and_data
        monkeypatch.setattr(feature_importance, "MODELS_DIR",  d["models_dir"])
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", d["figures_dir"])
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 10)
        monkeypatch.setattr(
            feature_importance, "load_and_split",
            lambda: (d["X_train"], d["X_train"], d["y_train"], d["y_train"])
        )
        plot_feature_importance()
        size = os.path.getsize(os.path.join(d["figures_dir"], "feature_importance.png"))
        assert size > 5000

    def test_prints_top_features(self, synthetic_rf_and_data, monkeypatch, capsys):
        """Output must include ranked features."""
        d = synthetic_rf_and_data
        monkeypatch.setattr(feature_importance, "MODELS_DIR",  d["models_dir"])
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", d["figures_dir"])
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 5)
        monkeypatch.setattr(
            feature_importance, "load_and_split",
            lambda: (d["X_train"], d["X_train"], d["y_train"], d["y_train"])
        )
        plot_feature_importance()
        captured = capsys.readouterr()
        assert "importance" in captured.out

    def test_prints_correct_number_of_features(self, synthetic_rf_and_data, monkeypatch, capsys):
        """Output must list exactly TOP_N_FEATURES features."""
        d = synthetic_rf_and_data
        monkeypatch.setattr(feature_importance, "MODELS_DIR",  d["models_dir"])
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", d["figures_dir"])
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 5)
        monkeypatch.setattr(
            feature_importance, "load_and_split",
            lambda: (d["X_train"], d["X_train"], d["y_train"], d["y_train"])
        )
        plot_feature_importance()
        captured = capsys.readouterr()
        # Each line has a rank number; count "1." through "5."
        lines_with_rank = [l for l in captured.out.splitlines() if "importance =" in l]
        assert len(lines_with_rank) == 5

    def test_falls_back_to_plain_rf_when_tuned_missing(self, synthetic_rf_and_data, monkeypatch, tmp_path):
        """If tuned RF is absent, should fall back to random_forest.joblib."""
        d = synthetic_rf_and_data
        # Remove tuned model
        tuned_path = os.path.join(d["models_dir"], "random_forest_tuned.joblib")
        if os.path.exists(tuned_path):
            os.remove(tuned_path)

        monkeypatch.setattr(feature_importance, "MODELS_DIR",  d["models_dir"])
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", d["figures_dir"])
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 5)
        monkeypatch.setattr(
            feature_importance, "load_and_split",
            lambda: (d["X_train"], d["X_train"], d["y_train"], d["y_train"])
        )
        # Should not raise
        plot_feature_importance()
        assert os.path.exists(os.path.join(d["figures_dir"], "feature_importance.png"))

    def test_no_model_returns_early(self, tmp_path, monkeypatch):
        """If no model file exists, function must return without crashing."""
        empty_models = tmp_path / "empty_models"
        empty_models.mkdir()
        figures_dir  = tmp_path / "figs"
        figures_dir.mkdir()

        monkeypatch.setattr(feature_importance, "MODELS_DIR",  str(empty_models))
        monkeypatch.setattr(feature_importance, "FIGURES_DIR", str(figures_dir))
        monkeypatch.setattr(feature_importance, "TOP_N_FEATURES", 5)
        # Should return early without raising
        plot_feature_importance()
        assert not os.path.exists(os.path.join(str(figures_dir), "feature_importance.png"))
