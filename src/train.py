"""
train.py — Train five models and save them to disk.

Models trained:
  1. DummyClassifier          — baseline: always predicts "not fraud"
  2. LogisticRegression       — fast linear model
  3. RandomForestClassifier   — ensemble, strong on tabular data
  4. XGBClassifier            — gradient boosting, often best on fraud data
  5. MLPClassifier            — simple feed-forward neural network

All real models handle the 1:578 class imbalance via class_weight='balanced'
(or scale_pos_weight for XGBoost), so the minority fraud class gets higher
weight during training without needing to resample the data.

After training the five base models, a RandomizedSearchCV is run on
Random Forest to demonstrate systematic hyperparameter tuning.

Cross-validation (5-fold, scoring=F1) is run on the training set before
we ever touch the held-out test data.
"""

import os
import sys
import logging
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    MODELS_DIR, CV_FOLDS, RANDOM_STATE,
    RF_N_ESTIMATORS, MLP_HIDDEN_LAYERS, MLP_MAX_ITER,
    LR_MAX_ITER, FRAUD_SCALE_POS_WEIGHT
)
from preprocessing import load_and_split

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_models():
    """Return a dict of untrained model instances."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),

        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # XGBoost uses scale_pos_weight instead of class_weight.
        # Setting it to ~578 (majority/minority ratio) has the same effect.
        "xgboost": XGBClassifier(
            scale_pos_weight=FRAUD_SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        ),

        # A two-hidden-layer MLP: 64 neurons → 32 neurons → output
        "neural_network": MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            random_state=RANDOM_STATE,
            early_stopping=True,      # stop if validation loss stops improving
            validation_fraction=0.1,
        ),
    }


def tune_random_forest(X_train, y_train):
    """
    Demonstrate hyperparameter tuning with RandomizedSearchCV on Random Forest.
    Searches 20 random combinations from the param grid and returns the best model.
    """
    log.info("Tuning Random Forest with RandomizedSearchCV (20 iterations) ...")

    param_dist = {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,              # 3-fold inside search to keep runtime reasonable
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    log.info(f"Best RF params : {search.best_params_}")
    log.info(f"Best CV F1     : {search.best_score_:.4f}")

    return search.best_estimator_


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_and_split()

    models = build_models()
    trained = {}

    log.info("\n── Cross-validation (F1, fraud class) ──────────────────────────")
    for name, model in models.items():
        log.info(f"Training '{name}' ...")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS, scoring="f1", n_jobs=-1
        )

        log.info(f"  CV F1 scores : {np.round(cv_scores, 4)}")
        log.info(f"  Mean F1      : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

        model.fit(X_train, y_train)

        out_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, out_path)
        log.info(f"  Saved → {out_path}")

        trained[name] = model

    # ── Hyperparameter tuning for Random Forest ───────────────────────────
    best_rf = tune_random_forest(X_train, y_train)
    tuned_path = os.path.join(MODELS_DIR, "random_forest_tuned.joblib")
    joblib.dump(best_rf, tuned_path)
    log.info(f"Tuned RF saved → {tuned_path}")
    trained["random_forest_tuned"] = best_rf

    log.info("\n── All models trained and saved ─────────────────────────────────\n")
    return trained, X_test, y_test


if __name__ == "__main__":
    train_all()
