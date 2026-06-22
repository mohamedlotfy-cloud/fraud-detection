"""
visualize.py — Generate all missing visualizations.

Figures produced (saved to reports/figures/):
  1. roc_curves.png              — ROC curves for all models on one plot
  2. correlation_heatmap.png     — feature correlation heatmap (sample of data)
  3. model_comparison_bar.png    — Precision / Recall / F1 bar chart per model
  4. tsne.png                    — t-SNE 2D projection coloured by fraud label

Run after train.py so the saved .joblib models exist.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    RANDOM_STATE, TSNE_SAMPLE_SIZE
)
from preprocessing import load_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Model names that evaluate.py also uses (excluding the tuned variant for clarity)
MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "xgboost", "neural_network"]

# Colour palette — one distinct colour per model
PALETTE = ["#999999", "#4C9BE8", "#2ECC71", "#E84C4C", "#9B59B6"]


def _load_models_and_data():
    """Load all saved models and the test split."""
    _, X_test, _, y_test = load_and_split()
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            log.warning(f"Model not found, skipping: {path}")
    return models, X_test, y_test


# ── Figure 1: ROC Curves ──────────────────────────────────────────────────────
def plot_roc_curves(models, X_test, y_test):
    """
    Plot all models' ROC curves on a single axes.
    ROC-AUC measures how well the model separates fraud from normal across
    all possible probability thresholds — 1.0 = perfect, 0.5 = random.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, model), color in zip(models.items(), PALETTE):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        label = f"{name.replace('_', ' ').title()}  (AUC = {roc_auc:.3f})"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    # Diagonal = random classifier
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out}")


# ── Figure 2: Correlation Heatmap ─────────────────────────────────────────────
def plot_correlation_heatmap():
    """
    Show the Pearson correlation between the 30 features.
    V1-V28 are PCA components so most are uncorrelated by design.
    Amount and Time may show weak correlations with some V-features.
    """
    from config.config import DATA_PATH
    df = pd.read_csv(DATA_PATH)

    # Use a subset of columns for readability — all 30 features + Class
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle (redundant)
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        vmin=-1, vmax=1, linewidths=0.3, annot=False,
        ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out}")


# ── Figure 3: Model Comparison Bar Chart ─────────────────────────────────────
def plot_model_comparison_bar(models, X_test, y_test):
    """
    Side-by-side bar chart comparing Precision, Recall, and F1 for every model.
    Makes it easy to spot trade-offs (e.g., high recall but low precision).
    """
    records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        records.append({
            "Model"    : name.replace("_", "\n"),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"   : recall_score(y_test, y_pred, zero_division=0),
            "F1"       : f1_score(y_test, y_pred, zero_division=0),
        })

    df = pd.DataFrame(records)
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    metric_colors = ["#4C9BE8", "#2ECC71", "#E84C4C"]

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric,
                      color=color, alpha=0.85, edgecolor="white")
        # Add value labels on top of each bar
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Precision / Recall / F1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "model_comparison_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out}")


# ── Figure 4: t-SNE ───────────────────────────────────────────────────────────
def plot_tsne(X_test, y_test):
    """
    t-SNE projects the 30-dimensional feature space down to 2D so we can
    visually check whether fraud and normal transactions form separate clusters.
    We sample TSNE_SAMPLE_SIZE rows because t-SNE is O(n²) and slow on the
    full 85k-row test set.
    """
    log.info(f"Running t-SNE on {TSNE_SAMPLE_SIZE:,} samples (this takes ~1–2 min) ...")

    # Sample equally from both classes to keep the plot balanced
    np.random.seed(RANDOM_STATE)
    X_arr = X_test.values
    y_arr = y_test.values

    fraud_idx  = np.where(y_arr == 1)[0]
    normal_idx = np.where(y_arr == 0)[0]

    n_fraud  = min(len(fraud_idx),  TSNE_SAMPLE_SIZE // 10)   # keep all fraud
    n_normal = TSNE_SAMPLE_SIZE - n_fraud

    chosen = np.concatenate([
        np.random.choice(fraud_idx,  n_fraud,  replace=False),
        np.random.choice(normal_idx, n_normal, replace=False),
    ])

    X_sample = X_arr[chosen]
    y_sample = y_arr[chosen]

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=300)
    X_2d = tsne.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {0: "#4C9BE8", 1: "#E84C4C"}
    labels = {0: f"Normal (n={n_normal:,})", 1: f"Fraud (n={n_fraud:,})"}

    for cls in [0, 1]:
        mask = y_sample == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[cls], label=labels[cls],
                   alpha=0.4, s=10, linewidths=0)

    ax.set_title("t-SNE Projection — Fraud vs Normal Transactions", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.legend(markerscale=3, fontsize=10)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "tsne.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    models, X_test, y_test = _load_models_and_data()

    if not models:
        log.error("No models found — run train.py first.")
        return

    log.info("Generating ROC curves ...")
    plot_roc_curves(models, X_test, y_test)

    log.info("Generating correlation heatmap ...")
    plot_correlation_heatmap()

    log.info("Generating model comparison bar chart ...")
    plot_model_comparison_bar(models, X_test, y_test)

    log.info("Generating t-SNE plot ...")
    plot_tsne(X_test, y_test)

    log.info("\n── All visualizations saved to reports/figures/ ─────────────────\n")


if __name__ == "__main__":
    run_all()
