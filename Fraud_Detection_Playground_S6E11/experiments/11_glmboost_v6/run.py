"""H16: GLM-Boost – LGBM (4 Leaves, wie lgbm_vte) mit init_score = GLM-Logit (glm_vte), lernt nur das Residuum.

Variante A: GLM-Logits für X_tr als innere 5-Fold-OOF, für Val/Test das GLM auf ganz X_tr.

    python experiments/11_glmboost_v6/run.py
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
    features="v6",          # Schlüssel aus src.features.FEATURE_SETS
    model="glmboost_vte",           # Schlüssel aus src.models.MODELS
    params={},              # überschreibt Default-Parameter des Modells (LGBM); glm_oof=True (Variante A) ist Default
    note="H16: GLM-Boost – LGBM L4 mit init_score = GLM-Logit (innere OOF, Variante A) auf v6-TE",
)

if __name__ == "__main__":
    import time
    t0 = time.time()
    train, test = load_data()
    X, X_test = features.build(CONFIG["features"], train, test)
    y = train[TARGET].values
    folds = get_folds(train)
    res = run_cv(models.get(CONFIG["model"], CONFIG["params"]), X, y, X_test, folds)
    save_experiment(NAME, res, test["id"], CONFIG["features"], CONFIG["model"],
                    params=CONFIG["params"], note=CONFIG["note"])
    print(f"LAUFZEIT {time.time() - t0:.0f} s")
