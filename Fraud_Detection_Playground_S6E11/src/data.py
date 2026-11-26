"""Laden der Rohdaten (CSV -> gecachtes Parquet) mit einheitlichen Dtypes."""
import pandas as pd

from .config import BOOL_COLS, CAT_COLS, DATA_DIR, ID, TARGET


def _read(name: str) -> pd.DataFrame:
    pq = DATA_DIR / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df.to_parquet(pq, index=False)
    return df


def _cast(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    for c in BOOL_COLS:
        df[c] = df[c].astype("int8")
    for c in CAT_COLS:
        # gleiche Kategorien in Train und Test -> konsistente Codes
        df[c] = pd.Categorical(df[c], categories=cats[c])
    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gibt (train, test) zurück; train enthält TARGET als int8."""
    train, test = _read("train"), _read("test")
    cats = {c: sorted(set(train[c].dropna()) | set(test[c].dropna())) for c in CAT_COLS}
    train = _cast(train, cats)
    test = _cast(test, cats)
    train[TARGET] = train[TARGET].astype("int8")
    return train, test


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")[[ID]]
