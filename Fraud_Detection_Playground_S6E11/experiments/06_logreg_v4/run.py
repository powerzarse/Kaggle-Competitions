"""06 – Logistische Regression (One-Hot + StandardScaler) auf Feature-Set v4 (H11).

Diverses, lineares Blend-Mitglied; C=1.0 per Fold-0-Vergleich (0.1/0.3/1.0) gewählt.

    python experiments/06_logreg_v4/run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import features, models  # noqa: E402
from src.config import TARGET  # noqa: E402
from src.cv import get_folds, run_cv  # noqa: E402
from src.data import load_data  # noqa: E402
from src.tracking import save_experiment  # noqa: E402

NAME = Path(__file__).resolve().parent.name   # = Ordnername, z.B. "02_xgb_v1"
CONFIG = dict(
    features="v4",          # Schlüssel aus src.features.FEATURE_SETS
    model="logreg",         # Schlüssel aus src.models.MODELS
    params={"C": 1.0},      # überschreibt Default-Parameter des Modells
    note="H11: LogReg auf v4 als diverses Blend-Mitglied",  # kurze Hypothese / Änderung für EXPERIMENTS.md
)

if __name__ == "__main__":
    train, test = load_data()
    X, X_test = features.build(CONFIG["features"], train, test)
    y = train[TARGET].values
    folds = get_folds(train)
    res = run_cv(models.get(CONFIG["model"], CONFIG["params"]), X, y, X_test, folds)
    save_experiment(NAME, res, test["id"], CONFIG["features"], CONFIG["model"],
                    params=CONFIG["params"], note=CONFIG["note"])
