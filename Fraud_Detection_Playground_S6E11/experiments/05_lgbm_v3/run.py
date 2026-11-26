"""05 – LightGBM mit fold-internem Target-Encoding (Kategorien + 3 Interaktionen), Feature-Set v3 (H06).

    python experiments/05_lgbm_v3/run.py
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
    features="v3",          # Schlüssel aus src.features.FEATURE_SETS
    model="lgbm_te",        # Schlüssel aus src.models.MODELS
    params={},              # überschreibt Default-Parameter des Modells
    note="H06: fold-internes Target-Encoding + 3 Interaktionen",  # kurze Hypothese / Änderung für EXPERIMENTS.md
)

if __name__ == "__main__":
    train, test = load_data()
    X, X_test = features.build(CONFIG["features"], train, test)
    y = train[TARGET].values
    folds = get_folds(train)
    res = run_cv(models.get(CONFIG["model"], CONFIG["params"]), X, y, X_test, folds)
    save_experiment(NAME, res, test["id"], CONFIG["features"], CONFIG["model"],
                    params=CONFIG["params"], note=CONFIG["note"])
