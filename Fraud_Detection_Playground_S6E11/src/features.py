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


# v3 (H06): Interaktions-Schlüssel als Rohmaterial für fold-internes Target-Encoding
# (src.models.lgbm_te). Die Schlüssel sind Strings und werden dort encodiert und entfernt.
V3_INTERACTIONS = [("channel", "device_type"), ("auth_method", "channel"),
                   ("merchant_category", "channel")]


def v3(df: pd.DataFrame) -> pd.DataFrame:
    """v1 + Interaktions-Schlüssel `<a>__<b>` (Strings, nur für Target-Encoding)."""
    X = v1(df)
    for a, b in V3_INTERACTIONS:
        X[f"{a}__{b}"] = df[a].astype(str).values + "|" + df[b].astype(str).values
    return X


FEATURE_SETS["v3"] = v3


def build(name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name in JOINT_FEATURE_SETS:
        return JOINT_FEATURE_SETS[name](train, test)
    fn = FEATURE_SETS[name]
    return fn(train), fn(test)


# v4 (H11): linearen-Modell-freundliche Variante von v1 (für LogReg). Bin-Grenzen sind feste
# Konstanten (gerundete Dezile über train+test, label-frei) -> kein Fitting, reproduzierbar.
V4_LOG_COLS = ["account_balance_usd", "distance_from_home_km", "hours_since_last_txn"]
V4_BINS = {
    "amount_usd": [9.22, 13.59, 18.12, 22.54, 27.25, 32.51, 39.08, 46.91, 91.08],
    "hours_since_last_txn": [0.85, 1.71, 2.54, 3.62, 4.72, 6.14, 7.94, 10.47, 15.0],
    "account_balance_usd": [593.62, 751.65, 912.6, 1138.62, 1385.2, 1677.01, 2214.87, 2899.29, 4026.04],
    "velocity_score": [0.05, 4.1, 9.5, 13.2, 15.8, 18.6, 21.4, 24.4, 28.0, 33.1],  # Bin 0 = genau 0
    "merchant_risk_score": [20.0, 24.7, 28.1, 31.2, 34.2, 37.3, 40.7, 45.1, 51.6],
    "customer_age": [25, 33, 41, 47, 52, 58, 63, 69, 76],
}
V4_CLIP_CATS = {"txn_count_last_24h": 7, "cvv_retry_count": 2, "prior_disputes": 3}
V4_FLAG_PAIRS = [("used_vpn", "ip_country_mismatch"), ("is_foreign_transaction", "ip_country_mismatch"),
                 ("is_foreign_transaction", "is_new_merchant"), ("is_new_merchant", "billing_shipping_mismatch"),
                 ("used_vpn", "is_new_merchant"), ("is_ai_generated_scam_attempt", "is_new_merchant")]


def v4(df: pd.DataFrame) -> pd.DataFrame:
    """v1 + velocity_is_zero, log1p schiefer Größen, feste Quantil-Bins als `category`,
    geclippte Klein-Zähler als `category`, Risiko-Flag-Interaktionen (nur für lineare Modelle)."""
    X = v1(df)
    X["velocity_is_zero"] = (df["velocity_score"] == 0).astype("int8")
    for c in V4_LOG_COLS:
        X[f"log_{c}"] = np.log1p(df[c])
    X["log_amount_to_balance"] = np.log1p(X["amount_to_balance"])
    X["log_txn_per_hour"] = np.log1p(X["txn_per_hour"])
    for c, edges in V4_BINS.items():
        X[f"{c}_bin"] = pd.Categorical(np.digitize(df[c], edges), categories=range(len(edges) + 1))
    for c, hi in V4_CLIP_CATS.items():
        X[f"{c}_lvl"] = pd.Categorical(df[c].clip(upper=hi), categories=range(hi + 1))
    for a, b in V4_FLAG_PAIRS:
        X[f"{a}_x_{b}"] = (df[a] * df[b]).astype("int8")
    X["cvv_retry_x_new_merchant"] = ((df["cvv_retry_count"] > 0) & (df["is_new_merchant"] == 1)).astype("int8")
    X["risk_flags_ge2"] = (X["risk_flag_sum"] >= 2).astype("int8")
    return X


FEATURE_SETS["v4"] = v4
