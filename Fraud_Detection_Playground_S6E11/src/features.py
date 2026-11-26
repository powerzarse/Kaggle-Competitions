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


FEATURE_SETS = {"base": base, "v1": v1}


def build(name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fn = FEATURE_SETS[name]
    return fn(train), fn(test)
