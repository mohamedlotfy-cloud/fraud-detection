"""
test_visualize.py — Unit tests for the plot functions in visualize.py

Each plot function is tested in isolation with synthetic data.
monkeypatch.setattr patches the module-level FIGURES_DIR and
TSNE_SAMPLE_SIZE variables directly in the visualize module.
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
import visualize   # import module to patch its namespace
from visualize import (
    plot_roc_curves,
    plot_model_comparison_bar,
    plot_tsne,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_xy():
    """300 samples, 30 features, ~15% fraud."""
    rng = np.random.default_rng(123)
    n = 300
    X = rng.standard_normal((n, 30))
    y = np.zeros(n, dtype=int)
    y[:45] = 1
    rng.shuffle(y)
    col_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    X_df = pd.DataFrame(X, columns=col_names)
    y_s  = pd.Series(y)
    return X_df, y_s


@pytest.fixture(scope="module")
def two_fitted_models(synthetic_xy):
    """Two quick classifiers fitted on synthetic data."""
    X, y = synthetic_xy
    dummy = DummyClassifier(strategy="stratified", random_state=0)
    dummy.fit(X, y)
    lr = LogisticRegression(max_iter=200, random_state=0)
    lr.fit(X, y)
    return {"dummy": dummy, "logistic_regression": lr}


# ── Tests: plot_roc_curves() ──────────────────────────────────────────────────

class TestPlotRocCurves:

    def test_creates_roc_png(self, two_fitted_models, synthetic_xy, tmp_path, monkeypatch):
        """plot_roc_curves must save roc_curves.png."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_roc_curves(two_fitted_models, X, y)
        assert os.path.exists(os.path.join(str(tmp_path), "roc_curves.png"))

    def test_roc_png_is_non_empty(self, two_fitted_models, synthetic_xy, tmp_path, monkeypatch):
        """roc_curves.png must be a real image file."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_roc_curves(two_fitted_models, X, y)
        size = os.path.getsize(os.path.join(str(tmp_path), "roc_curves.png"))
        assert size > 5000

    def test_roc_single_model(self, synthetic_xy, tmp_path, monkeypatch):
        """plot_roc_curves must work with just one model."""
        X, y = synthetic_xy
        lr = LogisticRegression(max_iter=200, random_state=0)
        lr.fit(X, y)
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_roc_curves({"logistic_regression": lr}, X, y)
        assert os.path.exists(os.path.join(str(tmp_path), "roc_curves.png"))


# ── Tests: plot_model_comparison_bar() ───────────────────────────────────────

class TestPlotModelComparisonBar:

    def test_creates_bar_chart_png(self, two_fitted_models, synthetic_xy, tmp_path, monkeypatch):
        """plot_model_comparison_bar must save model_comparison_bar.png."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_model_comparison_bar(two_fitted_models, X, y)
        assert os.path.exists(os.path.join(str(tmp_path), "model_comparison_bar.png"))

    def test_bar_chart_png_non_empty(self, two_fitted_models, synthetic_xy, tmp_path, monkeypatch):
        """model_comparison_bar.png must be a real image file."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_model_comparison_bar(two_fitted_models, X, y)
        size = os.path.getsize(os.path.join(str(tmp_path), "model_comparison_bar.png"))
        assert size > 5000

    def test_runs_with_single_model(self, synthetic_xy, tmp_path, monkeypatch):
        """bar chart must work with only one model."""
        X, y = synthetic_xy
        dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        plot_model_comparison_bar({"dummy": dummy}, X, y)
        assert os.path.exists(os.path.join(str(tmp_path), "model_comparison_bar.png"))


# ── Tests: plot_tsne() ────────────────────────────────────────────────────────

class TestPlotTsne:

    def test_creates_tsne_png(self, synthetic_xy, tmp_path, monkeypatch):
        """plot_tsne must create tsne.png."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        monkeypatch.setattr(visualize, "TSNE_SAMPLE_SIZE", 80)   # tiny → fast
        plot_tsne(X, y)
        assert os.path.exists(os.path.join(str(tmp_path), "tsne.png"))

    def test_tsne_png_non_empty(self, synthetic_xy, tmp_path, monkeypatch):
        """tsne.png must be a real image."""
        X, y = synthetic_xy
        monkeypatch.setattr(visualize, "FIGURES_DIR", str(tmp_path))
        monkeypatch.setattr(visualize, "TSNE_SAMPLE_SIZE", 80)
        plot_tsne(X, y)
        size = os.path.getsize(os.path.join(str(tmp_path), "tsne.png"))
        assert size > 5000
