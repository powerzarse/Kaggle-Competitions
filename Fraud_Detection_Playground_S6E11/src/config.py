"""Zentrale Konstanten für Playground Series S6E11 – Predicting Credit Card Fraud."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"          # OOF-/Test-Vorhersagen je Experiment (nicht im Git)
SUB_DIR = ROOT / "submissions"       # Submission-CSVs (nicht im Git)
EXP_DIR = ROOT / "experiments"
LOG_FILE = ROOT / "EXPERIMENTS.md"   # Ergebnis-Log (im Git)

COMPETITION = "playground-series-s6e11"
METRIC = "roc_auc"
ID = "id"
TARGET = "is_fraud"

SEED = 42
N_FOLDS = 5

NUM_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score",
    "prior_disputes",
]
CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
