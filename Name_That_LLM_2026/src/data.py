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


OPEN_POOL_PATH = DATA_DIR / "external" / "open_pool" / "ntl_open_pool.parquet"


def load_open_pool(per_class: int | None = None, kimi_reasoning_high_only: bool = False,
                   seed: int = 42) -> pd.DataFrame:
    """Externer Datensatz crtproduction/name-that-llm-open-pool (CC BY-SA 4.0, 224k Texte).

    Download:  kaggle datasets download crtproduction/name-that-llm-open-pool -p data/external/open_pool --unzip
    per_class: höchstens so viele Texte je Modell ziehen (balanciert, wichtig für Balanced Accuracy)
    kimi_reasoning_high_only: laut Datensatz-Autor ähneln nur kimi-k3-Texte mit Reasoning "high" den echten
    """
    cols = ["id", "text", "model", "prompt_group", "source_dataset", "gen_reasoning", "match_weight"]
    df = pd.read_parquet(OPEN_POOL_PATH, columns=cols)
    if kimi_reasoning_high_only:
        df = df[(df[TARGET] != "kimi-k3") | (df["gen_reasoning"] == "high")]
    if per_class:
        # mischen, dann die ersten per_class je Modell (pandas-3-sicher, ohne groupby.apply)
        df = df.sample(frac=1, random_state=seed).groupby(TARGET).head(per_class)
    df["family"] = df[TARGET].map(model_to_family)
    return df.reset_index(drop=True)
