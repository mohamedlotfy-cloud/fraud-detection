"""
demo.py — Quick demo: load the best model and run it on 5 random test rows.

Shows side-by-side: what the model predicted vs. what actually happened.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import MODELS_DIR
from preprocessing import load_and_split

BEST_MODEL = "random_forest_tuned"   # change if your results differ


def run_demo():
    _, X_test, _, y_test = load_and_split()

    # Fall back to base RF if tuned version doesn't exist yet
    model_path = os.path.join(MODELS_DIR, f"{BEST_MODEL}.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "random_forest.joblib")

    if not os.path.exists(model_path):
        print("ERROR: no model found — run train.py first.")
        return

    model = joblib.load(model_path)
    print(f"Loaded model: {os.path.basename(model_path)}\n")

    np.random.seed(7)
    sample_indices = np.random.choice(len(X_test), size=5, replace=False)
    X_sample       = X_test.iloc[sample_indices]
    y_sample       = y_test.iloc[sample_indices]

    predictions   = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    label = {0: "Normal", 1: "FRAUD "}

    print("── 5 Sample Predictions ─────────────────────────────────────────")
    print(f"{'#':<4} {'Actual':<10} {'Predicted':<10} {'P(fraud)':<10} {'Match?'}")
    print("-" * 50)

    for i, (pred, actual, prob) in enumerate(zip(predictions, y_sample, probabilities), 1):
        match = "✓" if pred == actual else "✗"
        print(f"{i:<4} {label[actual]:<10} {label[pred]:<10} {prob:<10.4f} {match}")

    print("\nNote: P(fraud) is the model's confidence that the transaction is fraudulent.")
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    run_demo()
