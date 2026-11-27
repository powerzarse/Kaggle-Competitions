"""Kompakter Lagebericht für Agenten: Rangliste, LB-Scores, Diversität, Budget, LB-Position.

    python -m src.status            # inkl. Kaggle-Abfrage (Limits, letzte Scores)
    python -m src.status --offline  # nur lokale Ergebnisse
"""
import json
import sys

import numpy as np

from .config import LOG_FILE, MODEL_DIR


def experiments() -> list[dict]:
    rows = [json.loads(p.read_text()) for p in MODEL_DIR.glob("*/meta.json")]
    return sorted(rows, key=lambda r: -r["cv"])


def lb_scores() -> dict[str, str]:
    """Experimentname -> Public LB aus EXPERIMENTS.md (Spalte 7)."""
    out = {}
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text().splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 7 and cells[1] and cells[6] and cells[6][0].isdigit():
                out[cells[1]] = cells[6]
    return out


def main(offline: bool = False) -> None:
    rows, lb = experiments(), lb_scores()
    if not rows:
        print("noch keine Experimente")
        return
    best = rows[0]["cv"]
    print(f"{'Experiment':<28}{'CV':>9}{'Δbest':>9}{'Std':>8}{'LB':>9}  Modell/Features")
    for r in rows:
        std = float(np.std(r["fold_scores"]))
        print(f"{r['name']:<28}{r['cv']:>9.5f}{r['cv'] - best:>+9.5f}{std:>8.4f}"
              f"{lb.get(r['name'], '-'):>9}  {r['model']}/{r['features']}")

    singles = [r["name"] for r in rows if not r["name"].startswith("blend")][:6]
    if len(singles) > 1:
        oof = np.column_stack([np.load(MODEL_DIR / n / "oof.npy") for n in singles])
        ranks = np.argsort(np.argsort(oof, axis=0), axis=0)
        corr = np.corrcoef(ranks.T)
        print("\nRang-Korrelation der OOFs (niedriger = diverser):")
        for i, n in enumerate(singles):
            print(f"  {n:<26}" + " ".join(f"{corr[i, j]:.3f}" for j in range(len(singles))))

    if not offline:
        from . import kaggle_api
        print()
        kaggle_api.limits()
        kaggle_api.submissions(5)
        print()
        kaggle_api.leaderboard(my_score=best)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
