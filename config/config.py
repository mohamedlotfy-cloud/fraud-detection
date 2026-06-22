# config.py — Central configuration for the fraud detection project.
# Import from here instead of hardcoding values in individual scripts.

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/raw/creditcard.csv"
MODELS_DIR   = "models"
FIGURES_DIR  = "reports/figures"
REPORTS_DIR  = "reports"

# ── Split & reproducibility ───────────────────────────────────────────────────
TEST_SIZE    = 0.30        # 70 % train / 30 % test
RANDOM_STATE = 42

# ── Cross-validation ──────────────────────────────────────────────────────────
CV_FOLDS     = 5

# ── Model hyperparameters ─────────────────────────────────────────────────────
RF_N_ESTIMATORS   = 100
MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER      = 200
LR_MAX_ITER       = 1000

# ── Imbalance ratio (used by XGBoost scale_pos_weight) ───────────────────────
# 284315 normal / 492 fraud ≈ 578
FRAUD_SCALE_POS_WEIGHT = 578

# ── Visualisation ─────────────────────────────────────────────────────────────
TSNE_SAMPLE_SIZE = 5_000   # t-SNE is slow — run on a random sample only
TOP_N_FEATURES   = 10
