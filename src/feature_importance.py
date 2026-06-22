"""
feature_importance.py — Plot the top N most important features from
the trained Random Forest model.

Random Forests measure feature importance as the average reduction in
impurity (Gini impurity) across all trees for each feature.
Higher = that feature caused more splits separating fraud from normal.
"""

import os
import sys
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import MODELS_DIR, FIGURES_DIR, TOP_N_FEATURES
from preprocessing import load_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def plot_feature_importance():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Prefer the tuned RF if available
    for model_name in ["random_forest_tuned", "random_forest"]:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if os.path.exists(model_path):
            break
    else:
        log.error("No Random Forest model found — run train.py first.")
        return

    rf = joblib.load(model_path)
    log.info(f"Loaded model: {model_name}")

    X_train, _, _, _ = load_and_split()
    feature_names = X_train.columns.tolist()

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1]

    top_indices = indices[:TOP_N_FEATURES]
    top_names   = [feature_names[i] for i in top_indices]
    top_scores  = importances[top_indices]

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, TOP_N_FEATURES))[::-1]
    ax.barh(range(TOP_N_FEATURES), top_scores[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(TOP_N_FEATURES))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Mean Decrease in Impurity (feature importance)", fontsize=10)
    ax.set_title(f"Top {TOP_N_FEATURES} Most Important Features — Random Forest",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Feature importance plot saved → {out_path}")

    print(f"\nTop {TOP_N_FEATURES} features ranked:")
    for rank, (name, score) in enumerate(zip(top_names, top_scores), 1):
        print(f"  {rank:2d}. {name:10s}  importance = {score:.4f}")


if __name__ == "__main__":
    plot_feature_importance()
