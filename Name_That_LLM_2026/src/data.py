"""Laden der Wettbewerbsdaten und Ableitung der Modellfamilie."""
import pandas as pd

from src.config import DATA_DIR

TARGET = "model"
TEXT_COL = "text"
GROUP_COL = "prompt_id"

FAMILY_PREFIXES = ("claude", "gpt", "gemini", "glm", "qwen", "kimi")


def model_to_family(model: str) -> str:
    """'claude-opus-4-6' -> 'claude', 'qwen3.8-max' -> 'qwen'."""
    for prefix in FAMILY_PREFIXES:
        if model.startswith(prefix):
            return prefix
    return "other"


def load_train() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["family"] = df[TARGET].map(model_to_family)
    return df


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "sample_submission.parquet")
