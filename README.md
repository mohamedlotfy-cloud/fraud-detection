# Credit Card Fraud Detection — ML Project

A machine-learning project that classifies credit card transactions as fraud or not,
using the public ULB / Kaggle dataset (`creditcard.csv`).

The dataset contains **284,807 transactions** with 30 features (28 anonymized PCA
components plus `Amount` and `Time`) and is highly imbalanced — only ~0.17% of
transactions are fraudulent.

**Five models** are trained and compared, with hyperparameter tuning applied to
the best performer. Evaluation focuses on Precision, Recall, and F1 rather than
raw accuracy, since a model that always predicts "normal" would be 99.8% accurate
while catching zero fraud.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it at:

```
data/raw/creditcard.csv
```

---

## How to Run (in order)

### 1. Explore the data
```bash
python src/eda.py
```
Prints class distribution and saves 2 figures to `reports/figures/`.

### 2. Train all models
```bash
python src/train.py
```
Trains 5 models + tunes Random Forest with RandomizedSearchCV.
Saves `.joblib` files to `models/`.

### 3. Evaluate on the test set
```bash
python src/evaluate.py
```
Computes Precision, Recall, F1, ROC-AUC per model.
Saves confusion matrices and `reports/model_comparison.csv`.

### 4. Generate remaining visualizations
```bash
python src/visualize.py
```
Saves ROC curves, correlation heatmap, model comparison bar chart, and t-SNE plot.

### 5. Feature importance
```bash
python src/feature_importance.py
```
Plots top 10 RF features to `reports/figures/feature_importance.png`.

### 6. Quick demo
```bash
python src/demo.py
```
Loads the best model and runs it on 5 random test rows.

### 7. Run unit tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Results Summary

| Model                  | Precision | Recall | F1     | ROC-AUC |
|------------------------|-----------|--------|--------|---------|
| Dummy                  | 0.0000    | 0.0000 | 0.0000 | ~0.500  |
| Logistic Regression    | ~0.87     | ~0.62  | ~0.72  | ~0.97   |
| Random Forest          | ~0.92     | ~0.80  | ~0.86  | ~0.98   |
| XGBoost                | ~0.93     | ~0.82  | ~0.87  | ~0.98   |
| Neural Network (MLP)   | ~0.88     | ~0.75  | ~0.81  | ~0.97   |
| Random Forest (tuned)  | ~0.93     | ~0.83  | ~0.88  | ~0.98   |

> Exact numbers vary by run. XGBoost and tuned RF consistently lead.

---

## Project Structure

```
fraud-detection/
├── config/
│   └── config.py               ← all constants & paths (no magic numbers in code)
├── data/
│   ├── raw/creditcard.csv      ← place dataset here
│   └── processed/              ← reserved for future outputs
├── src/
│   ├── eda.py                  ← Step 1: exploratory analysis
│   ├── preprocessing.py        ← shared: load + split + scale
│   ├── train.py                ← Step 2: train 5 models + hyperparameter tuning
│   ├── evaluate.py             ← Step 3: test-set metrics + confusion matrices
│   ├── visualize.py            ← Step 4: ROC curves, heatmap, bar chart, t-SNE
│   ├── feature_importance.py   ← Step 5: RF feature importance plot
│   └── demo.py                 ← Step 6: 5-row live predictions
├── models/                     ← saved .joblib model files
├── reports/
│   ├── figures/                ← all plots (8+ figures)
│   └── model_comparison.csv    ← metrics table
├── tests/
│   ├── test_preprocessing.py   ← 13 unit tests for preprocessing
│   ├── test_models.py          ← 15 unit tests for model training & tuning
│   ├── test_evaluate.py        ← 10 unit tests for evaluation logic
│   ├── test_eda.py             ← 7 unit tests for EDA
│   ├── test_visualize.py       ← 8 unit tests for visualisation
│   └── test_feature_importance.py ← 6 unit tests for feature importance
├── notebooks/                  ← reserved for Jupyter EDA notebooks
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Class imbalance**: handled via `class_weight='balanced'` (sklearn) and
  `scale_pos_weight` (XGBoost), without resampling. This avoids information
  leakage that can occur with SMOTE if applied before splitting.
- **Data leakage prevention**: StandardScaler is fit only on the training set;
  the same fitted scaler transforms the test set.
- **Stratified split**: guarantees the rare fraud class appears in both train
  and test with the same ratio as the full dataset.
- **Evaluation metric**: F1 score (not accuracy) throughout — accuracy is
  misleading on imbalanced datasets.
