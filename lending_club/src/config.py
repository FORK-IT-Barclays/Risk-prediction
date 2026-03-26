"""
config.py — Central configuration for the Financial Historian Model
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / 'data' / 'processed'
MODEL_DIR = ROOT / 'outputs' / 'models'
PLOT_DIR  = ROOT / 'outputs' / 'plots'
REPORT_DIR= ROOT / 'reports'

for d in [MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PARQUET_PATH       = DATA_DIR / 'lending_club_preprocessed.parquet'
CLEAN_PARQUET_PATH = DATA_DIR / 'lending_club_clean.parquet'   # ← post-cleaning
CSV_PATH           = DATA_DIR / 'lending_club_preprocessed_sample.csv'

# ── Feature splits ───────────────────────────────────────────────────────────
# T=0 — available at loan origination (no leakage)
T0_FEATURES = [
    'loan_amnt', 'term_months', 'int_rate',
    'grade_enc', 'sub_grade_enc',
    'annual_inc', 'dti',
    'fico_avg',
    'delinq_2yrs', 'pub_rec', 'inq_last_6mths',
    'open_acc', 'total_acc',
    'revol_bal', 'revol_util',
    'cr_history_months',
    'loan_to_income', 'inst_to_income',
    'delinq_recency_score', 'inq_pressure',
]

# T>0 — behavioural signals (available mid-loan, not post-default)
T_BEHAVIORAL = [
    'fico_drop',        # change in FICO since origination
    'last_fico_avg',    # latest FICO reading
    'repay_ratio',      # principal paid / loan_amnt
    'out_prncp_ratio',  # outstanding principal / loan_amnt
    'late_fee_flag',    # any late fee ever recorded (binary)
]

# EXCLUDED — post-default leakage features
EXCLUDED_FEATURES = [
    'recoveries',           # set only after charge-off
    'total_rec_late_fee',   # raw amount (late_fee_flag is the safe binary form)
]

# Full historian feature set (T0 + behavioural)
ALL_FEATURES = T0_FEATURES + T_BEHAVIORAL

TARGET = 'risk_label'

# ── Temporal split ────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2012, 2017))   # 2012–2016: post-GFC training window
VAL_YEAR    = 2017
TEST_YEAR   = 2018

# ── Model params ──────────────────────────────────────────────────────────────
XGB_PARAMS = {
    'n_estimators':      500,
    'max_depth':         6,
    'learning_rate':     0.05,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'min_child_weight':  10,
    'eval_metric':       'aucpr',
    'use_label_encoder': False,
    'random_state':      42,
    'n_jobs':            -1,
}

DECISION_THRESHOLD = 0.30   # lower than default 0.5 for higher recall

SEED = 42
