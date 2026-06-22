"""
eda.py — Exploratory Data Analysis for Credit Card Fraud Detection

Run this first to understand the dataset before training anything.
Figures saved to reports/figures/.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import DATA_PATH, FIGURES_DIR


def run_eda(path=DATA_PATH):
    # ── Load & basic sanity check ──────────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}  ✓" if missing == 0 else f"WARNING: {missing} missing values found!")

    # ── Class distribution ─────────────────────────────────────────────────
    fraud_count  = df["Class"].sum()
    normal_count = len(df) - fraud_count
    fraud_pct    = 100 * fraud_count / len(df)
    print(f"\nClass breakdown:")
    print(f"  Normal transactions : {normal_count:,}")
    print(f"  Fraud  transactions : {fraud_count:,}  ({fraud_pct:.3f}% of total)")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Figure 1: Bar chart — fraud vs normal counts ───────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(["Normal (0)", "Fraud (1)"], counts.values,
                  color=["#4C9BE8", "#E84C4C"], edgecolor="white", linewidth=1.2)
    ax.set_title("Transaction Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Transactions")
    ax.set_xlabel("Class")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.annotate("Note: log scale used\ndue to severe imbalance",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="gray")
    plt.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "class_distribution.png")
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # ── Figure 2: Histogram — transaction amount by class ──────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    normal_amounts = df.loc[df["Class"] == 0, "Amount"]
    fraud_amounts  = df.loc[df["Class"] == 1, "Amount"]

    ax.hist(normal_amounts.clip(upper=2000), bins=80, alpha=0.6,
            label=f"Normal (n={normal_count:,})", color="#4C9BE8", density=True)
    ax.hist(fraud_amounts.clip(upper=2000),  bins=80, alpha=0.8,
            label=f"Fraud  (n={fraud_count:,})", color="#E84C4C", density=True)
    ax.set_title("Transaction Amount Distribution — Fraud vs Normal", fontsize=13, fontweight="bold")
    ax.set_xlabel("Amount (€, clipped at 2 000)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "amount_distribution.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # ── Printed summary ────────────────────────────────────────────────────
    print("\n── EDA Summary ──────────────────────────────────────────────────")
    print(
        f"The dataset is severely imbalanced: only {fraud_pct:.3f}% of transactions "
        f"({fraud_count:,} out of {len(df):,}) are fraudulent. "
        "Standard accuracy is therefore a poor metric — a model that always predicts 'normal' "
        "would score 99.8% accuracy while catching zero fraud cases. "
        "We'll use Precision, Recall, and F1 score instead, and apply class-weight balancing during training."
    )
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    run_eda()
