"""
evaluate.py — Evaluate all saved models on the held-out test set.

Why not use accuracy?
    On this dataset, always predicting "not fraud" gives ~99.83% accuracy —
    while catching exactly zero fraud cases.  Precision, Recall, and F1 tell
    a much more honest story for imbalanced classification.

Outputs:
  - reports/figures/{model_name}_confusion_matrix.png  (one per model)
  - reports/model_comparison.csv                       (all metrics in one table)
  - Printed examples: fraud cases caught + missed by the best model
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from preprocessing import load_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_NAMES = ["dummy", "logistic_regression", "random_forest",
               "xgboost", "neural_network", "random_forest_tuned"]


def evaluate_model(name, model, X_test, y_test):
    """Return a dict of metrics and save the confusion-matrix heatmap."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # ── Confusion matrix heatmap ───────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                xticklabels=["Pred: Normal", "Pred: Fraud"],
                yticklabels=["True: Normal", "True: Fraud"])
    ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, f"{name}_confusion_matrix.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log.info(f"  Confusion matrix saved → {fig_path}")

    return {
        "Model"    : name,
        "Precision": round(precision, 4),
        "Recall"   : round(recall, 4),
        "F1"       : round(f1, 4),
        "ROC-AUC"  : round(roc_auc, 4),
    }


def print_example_cases(name, model, X_test, y_test):
    """Print fraud cases the model caught and cases it missed."""
    y_pred   = model.predict(X_test)
    X_arr    = X_test.values if hasattr(X_test, "values") else X_test
    y_true   = np.array(y_test)

    caught_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    missed_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    print(f"\n── Example cases for '{name}' ───────────────────────────────────")
    print("  3 fraud cases CAUGHT (true positive):")
    for i, idx in enumerate(caught_idx[:3], 1):
        amount = X_arr[idx, X_test.columns.get_loc("Amount")]
        print(f"    [{i}] Row {idx} — scaled amount={amount:.2f}  |  True=Fraud  Pred=Fraud  ✓")

    print("  3 fraud cases MISSED (false negative):")
    for i, idx in enumerate(missed_idx[:3], 1):
        amount = X_arr[idx, X_test.columns.get_loc("Amount")]
        print(f"    [{i}] Row {idx} — scaled amount={amount:.2f}  |  True=Fraud  Pred=Normal  ✗")
    print()


def run_evaluation():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    _, X_test, _, y_test = load_and_split()

    results = []
    log.info("\n── Evaluating models on held-out test set ───────────────────────")

    for name in MODEL_NAMES:
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.exists(model_path):
            log.warning(f"  SKIP: {model_path} not found — run train.py first.")
            continue

        log.info(f"\n  Model: {name}")
        model   = joblib.load(model_path)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        log.info(f"  Precision : {metrics['Precision']:.4f}")
        log.info(f"  Recall    : {metrics['Recall']:.4f}")
        log.info(f"  F1        : {metrics['F1']:.4f}")
        log.info(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")

    comparison_df = pd.DataFrame(results)
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    log.info(f"\n── Model comparison saved → {csv_path}")
    print(comparison_df.to_string(index=False))

    best_row   = comparison_df.loc[comparison_df["F1"].idxmax()]
    best_name  = best_row["Model"]
    log.info(f"\n── Best model by F1: '{best_name}' (F1={best_row['F1']:.4f})")
    best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_name}.joblib"))
    print_example_cases(best_name, best_model, X_test, y_test)


if __name__ == "__main__":
    run_evaluation()
