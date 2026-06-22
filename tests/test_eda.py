"""
test_eda.py — Unit tests for eda.py

Tests run_eda() with a synthetic CSV — no real creditcard.csv needed.
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

import eda   # import the module so we can patch its namespace
from eda import run_eda


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_csv(tmp_path):
    """Create a tiny creditcard-style CSV with 500 rows and 5% fraud."""
    rng = np.random.default_rng(7)
    n = 500
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Amount"] = rng.uniform(0, 2000, n)
    data["Time"]   = np.arange(n, dtype=float)
    fraud_idx = rng.choice(n, size=25, replace=False)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][fraud_idx] = 1

    df = pd.DataFrame(data)
    path = tmp_path / "creditcard.csv"
    df.to_csv(path, index=False)
    return str(path), df


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunEda:

    def test_runs_without_error(self, synthetic_csv, tmp_path, monkeypatch):
        """run_eda must complete without raising an exception."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)

    def test_creates_class_distribution_figure(self, synthetic_csv, tmp_path, monkeypatch):
        """run_eda must produce class_distribution.png."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        assert os.path.exists(os.path.join(str(tmp_path), "class_distribution.png"))

    def test_creates_amount_distribution_figure(self, synthetic_csv, tmp_path, monkeypatch):
        """run_eda must produce amount_distribution.png."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        assert os.path.exists(os.path.join(str(tmp_path), "amount_distribution.png"))

    def test_prints_row_count(self, synthetic_csv, tmp_path, capsys, monkeypatch):
        """run_eda must print the number of rows loaded."""
        path, df = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        captured = capsys.readouterr()
        assert str(len(df)) in captured.out

    def test_prints_no_missing_values_message(self, synthetic_csv, tmp_path, capsys, monkeypatch):
        """run_eda must report zero missing values for a clean dataset."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        captured = capsys.readouterr()
        assert "Missing values: 0" in captured.out

    def test_prints_fraud_percentage(self, synthetic_csv, tmp_path, capsys, monkeypatch):
        """run_eda must print a fraud percentage."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        captured = capsys.readouterr()
        assert "%" in captured.out

    def test_figures_are_valid_png_files(self, synthetic_csv, tmp_path, monkeypatch):
        """Saved PNG files must be non-empty."""
        path, _ = synthetic_csv
        monkeypatch.setattr(eda, "FIGURES_DIR", str(tmp_path))
        run_eda(path=path)
        for fname in ["class_distribution.png", "amount_distribution.png"]:
            fpath = os.path.join(str(tmp_path), fname)
            assert os.path.getsize(fpath) > 1000, f"{fname} is suspiciously small"
