"""Modell-Factories im fit_predict-Format von src.cv.run_cv.

XGBoost/CatBoost werden erst beim Aufruf importiert -> optional installieren.
"""
from .config import SEED


def lgbm(params: dict | None = None, early_stopping: int = 200):
    import lightgbm as lgb

    p = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
             n_estimators=5000, random_state=SEED, verbose=-1)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]

    return fit_predict


def xgb(params: dict | None = None, early_stopping: int = 200):
    import xgboost as xgb_

    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03, max_depth=6,
             subsample=0.8, colsample_bytree=0.7, min_child_weight=5, n_estimators=5000,
             tree_method="hist", enable_categorical=True, random_state=SEED,
             early_stopping_rounds=early_stopping)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        m = xgb_.XGBClassifier(**p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]

    return fit_predict


def catboost(params: dict | None = None, early_stopping: int = 200):
    from catboost import CatBoostClassifier

    p = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05, depth=6,
             iterations=5000, random_seed=SEED, verbose=0, od_type="Iter", od_wait=early_stopping)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
        conv = lambda X: X.astype({c: str for c in cats})  # noqa: E731
        m = CatBoostClassifier(**p, cat_features=cats)
        m.fit(conv(X_tr), y_tr, eval_set=(conv(X_va), y_va))
        return m.predict_proba(conv(X_va))[:, 1], m.predict_proba(conv(X_te))[:, 1]

    return fit_predict


def logreg(params: dict | None = None):
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        cats = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
        nums = [c for c in X_tr.columns if c not in cats]
        pre = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cats),
                                      (StandardScaler(), nums))
        m = make_pipeline(pre, LogisticRegression(**{"C": 1.0, "max_iter": 2000, **(params or {})}))
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]

    return fit_predict


MODELS = {"lgbm": lgbm, "xgb": xgb, "catboost": catboost, "logreg": logreg}


def get(name: str, params: dict | None = None):
    """Jede Factory akzeptiert `params` (überschreibt die Defaults)."""
    return MODELS[name](params=params)

