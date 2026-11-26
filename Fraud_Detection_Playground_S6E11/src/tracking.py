"""Speichert Experiment-Ergebnisse einheitlich und führt das Log in EXPERIMENTS.md.

Ablage je Experiment:
    models/<name>/oof.npy, test.npy, meta.json
    submissions/<name>.csv
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from .config import ID, LOG_FILE, MODEL_DIR, SUB_DIR, TARGET

LOG_HEADER = (
    "| Datum | Experiment | Features | Modell | CV AUC (OOF) | Fold-Std | Public LB | Notiz |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def save_experiment(name: str, result: dict, test_ids, features: str, model: str,
                    params: dict | None = None, note: str = "") -> None:
    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", result["oof"])
    np.save(out / "test.npy", result["test"])
    meta = dict(name=name, features=features, model=model, params=params or {},
                cv=result["cv"], fold_scores=result["fold_scores"], date=str(date.today()))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    SUB_DIR.mkdir(exist_ok=True)
    sub = pd.DataFrame({ID: np.asarray(test_ids), TARGET: result["test"]})
    sub.to_csv(SUB_DIR / f"{name}.csv", index=False)

    append_log(name, features, model, result["cv"], float(np.std(result["fold_scores"])), note)
    print(f"gespeichert: models/{name}/ und submissions/{name}.csv")


def append_log(name, features, model, cv, std, note="", lb="") -> None:
    if not LOG_FILE.exists():
        LOG_FILE.write_text("# Experiment-Log\n\n" + LOG_HEADER)
    row = f"| {date.today()} | {name} | {features} | {model} | {cv:.5f} | {std:.5f} | {lb} | {note} |\n"
    with LOG_FILE.open("a") as f:
        f.write(row)


def load_oof(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(oof, test) als Matrizen [n_rows, n_models] für Blending."""
    oof = np.column_stack([np.load(MODEL_DIR / n / "oof.npy") for n in names])
    test = np.column_stack([np.load(MODEL_DIR / n / "test.npy") for n in names])
    return oof, test


def available_experiments() -> list[str]:
    return sorted(p.name for p in MODEL_DIR.iterdir() if (p / "oof.npy").exists())
