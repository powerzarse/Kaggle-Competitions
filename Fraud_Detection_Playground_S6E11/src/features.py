"""Feature-Sets. Jedes Experiment wählt per Name ein Set -> reproduzierbar & vergleichbar."""
import numpy as np
import pandas as pd

from .config import BOOL_COLS, CAT_COLS, NUM_COLS


def base(df: pd.DataFrame) -> pd.DataFrame:
    """Nur Originalspalten."""
    return df[NUM_COLS + BOOL_COLS + CAT_COLS].copy()


def v1(df: pd.DataFrame) -> pd.DataFrame:
    """Originalspalten + einfache Domänen-Features."""
    X = base(df)
    X["log_amount"] = np.log1p(df["amount_usd"])
    X["amount_to_balance"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
    X["risk_flag_sum"] = df[BOOL_COLS].sum(axis=1)
    X["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24)
    X["is_night"] = df["time_of_day_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    X["txn_per_hour"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 0.1)
    X["amount_cents"] = (df["amount_usd"] * 100 % 100).round().astype("int16")
    return X


# v2 (H05): Playground-Rundungsartefakte -> exakte Werte / Häufigkeiten als Information
V2_CAT_COLS = ["txn_count_last_24h", "time_of_day_hour", "card_age_months", "customer_age",
               "velocity_score", "merchant_risk_score"]
V2_CNT_COLS = ["velocity_score", "merchant_risk_score", "hours_since_last_txn",
               "distance_from_home_km", "amount_usd", "account_balance_usd"]


def v2(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v1 + numerische Spalten als `category` (_cat) + Count-Encoding (_cnt).

    Sieht train und test gemeinsam: Kategorienliste und Häufigkeiten werden über train+test
    bestimmt. Es fließen nur Werte-Häufigkeiten ein, keine Labels -> kein Target-Leakage.
    """
    X, X_te = v1(train), v1(test)
    for c in V2_CAT_COLS:
        cats = np.sort(pd.unique(pd.concat([train[c], test[c]], ignore_index=True)))
        X[f"{c}_cat"] = pd.Categorical(train[c], categories=cats)
        X_te[f"{c}_cat"] = pd.Categorical(test[c], categories=cats)
    for c in V2_CNT_COLS:
        counts = pd.concat([train[c], test[c]], ignore_index=True).value_counts()
        X[f"{c}_cnt"] = train[c].map(counts).astype("int32").values
        X_te[f"{c}_cnt"] = test[c].map(counts).astype("int32").values
    return X, X_te


FEATURE_SETS = {"base": base, "v1": v1}
# Sets, die train und test gemeinsam sehen müssen (nur label-freie Statistiken!)
JOINT_FEATURE_SETS = {"v2": v2}


def build(name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name in JOINT_FEATURE_SETS:
        return JOINT_FEATURE_SETS[name](train, test)
    fn = FEATURE_SETS[name]
    return fn(train), fn(test)
