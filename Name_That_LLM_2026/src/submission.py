"""Submission im Format der sample_submission.parquet (id, model)."""
from datetime import date

import numpy as np
import pandas as pd

from src.config import SUBMISSION_DIR


def make_submission(ids, proba: np.ndarray, classes, name: str) -> pd.DataFrame:
    sub = pd.DataFrame({"id": ids, "model": np.asarray(classes)[proba.argmax(1)]})
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"submission_{name}_{date.today():%Y%m%d}.parquet"
    sub.to_parquet(path, index=False)
    print(f"gespeichert: {path}")
    return sub
