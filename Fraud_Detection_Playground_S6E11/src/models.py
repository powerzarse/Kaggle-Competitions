"""Modell-Factories im fit_predict-Format von src.cv.run_cv.

XGBoost/CatBoost werden erst beim Aufruf importiert -> optional installieren.
"""
import numpy as np
import pandas as pd

from .config import CAT_COLS, SEED


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


def _te_fit(keys, y, m, prior):
    """m-Estimate-Mittel je Schlüssel: (sum_y + m*prior) / (n + m)."""
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k", sort=False)["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


def target_encode(X_tr, y_tr, X_va, X_te, cols, m=20.0, n_inner=5, seed=SEED):
    """Leakage-freies Target-Encoding der Spalten `cols` -> neue Spalten `<c>_te`.

    X_tr: out-of-fold über innere StratifiedKFold-Splits (nur auf X_tr/y_tr).
    X_va, X_te: Mapping aus ganz X_tr. Unbekannte Werte -> globaler Mittelwert von y_tr.
    """
    from sklearn.model_selection import StratifiedKFold

    y_tr = np.asarray(y_tr, dtype=float)
    prior = y_tr.mean()
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y_tr)), y_tr))
    for c in cols:
        k_tr = X_tr[c].astype(str).values
        oof = np.empty(len(y_tr))
        for i_fit, i_enc in inner:
            p_in = y_tr[i_fit].mean()
            mp = _te_fit(k_tr[i_fit], y_tr[i_fit], m, p_in)
            oof[i_enc] = pd.Series(k_tr[i_enc]).map(mp).fillna(p_in).values
        mp = _te_fit(k_tr, y_tr, m, prior)
        X_tr[f"{c}_te"] = oof
        X_va[f"{c}_te"] = pd.Series(X_va[c].astype(str).values).map(mp).fillna(prior).values
        X_te[f"{c}_te"] = pd.Series(X_te[c].astype(str).values).map(mp).fillna(prior).values
    return X_tr, X_va, X_te


def lgbm_te(params: dict | None = None, early_stopping: int = 200, m: float = 20.0, n_inner: int = 5):
    """Fold-internes Target-Encoding (H06), danach exakt `lgbm` (gleiche Defaults/Early Stopping).

    Encodiert CAT_COLS (bleiben zusätzlich als `category`) und alle Interaktions-Schlüssel
    (String-Spalten mit `__` im Namen, z. B. aus Feature-Set v3); die Schlüssel werden danach entfernt.
    """
    base_fp = lgbm(params, early_stopping)

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        keys = [c for c in X_tr.columns if "__" in c and pd.api.types.is_string_dtype(X_tr[c])
                and not isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
        cols = [c for c in CAT_COLS if c in X_tr.columns] + keys
        X_tr, X_va, X_te = target_encode(X_tr, y_tr, X_va, X_te, cols, m=m, n_inner=n_inner)
        X_tr, X_va, X_te = (X.drop(columns=keys) for X in (X_tr, X_va, X_te))
        return base_fp(X_tr, y_tr, X_va, y_va, X_te)

    return fit_predict


MODELS = {"lgbm": lgbm, "xgb": xgb, "catboost": catboost, "logreg": logreg}
MODELS["lgbm_te"] = lgbm_te


def get(name: str, params: dict | None = None):
    """Jede Factory akzeptiert `params` (überschreibt die Defaults)."""
    return MODELS[name](params=params)



def target_encode_keys(X_tr, y_tr, X_va, X_te, cols, m=20.0, n_inner=5, seed=SEED):
    """Vektorisierte Variante von `target_encode` (gleiche Semantik) für viele hochkardinale Schlüssel.

    Gibt drei DataFrames mit den Spalten `<c>_te` zurück (float32). Codes per pd.factorize über
    X_tr/X_va/X_te gemeinsam (label-frei), Raten per np.bincount.
    X_tr: OOF über innere StratifiedKFold-Splits (Prior = Mittel des inneren Fit-Teils);
    X_va/X_te: Mapping aus ganz X_tr (Prior = Mittel von y_tr); unbekannte Werte -> Prior.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y_tr, dtype=float)
    n_tr, n_va = len(X_tr), len(X_va)
    prior = y.mean()
    inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(n_tr), y))
    out = {k: {} for k in ("tr", "va", "te")}

    def rate(codes, yy, n, p):
        return (np.bincount(codes, weights=yy, minlength=n) + m * p) / (np.bincount(codes, minlength=n) + m)

    for c in cols:
        codes = pd.factorize(np.concatenate([X_tr[c].values, X_va[c].values, X_te[c].values]))[0]
        n = codes.max() + 1
        k_tr, k_va, k_te = codes[:n_tr], codes[n_tr:n_tr + n_va], codes[n_tr + n_va:]
        oof = np.empty(n_tr)
        for i_fit, i_enc in inner:
            # ungesehene Werte haben count 0 -> rate = Prior des inneren Fit-Teils
            oof[i_enc] = rate(k_tr[i_fit], y[i_fit], n, y[i_fit].mean())[k_tr[i_enc]]
        full = rate(k_tr, y, n, prior)  # Werte, die nur in Val/Test vorkommen -> Prior
        out["tr"][f"{c}_te"] = oof.astype(np.float32)
        out["va"][f"{c}_te"] = full[k_va].astype(np.float32)
        out["te"][f"{c}_te"] = full[k_te].astype(np.float32)
    return tuple(pd.DataFrame(out[k], index=X.index) for k, X in (("tr", X_tr), ("va", X_va), ("te", X_te)))


def vte_frames(X_tr, y_tr, X_va, X_te, m: float = 20.0, n_inner: int = 5):
    """Gemeinsamer TE-Schritt von lgbm_vte/xgb_vte/glm_vte: alle `__key`-Spalten (Strings) fold-intern per
    `target_encode_keys` encodieren, Schlüssel entfernen, `<c>_te` anhängen. -> (X_tr, X_va, X_te, keys)."""
    keys = [c for c in X_tr.columns if c.endswith("__key") and pd.api.types.is_string_dtype(X_tr[c])
            and not isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
    E_tr, E_va, E_te = target_encode_keys(X_tr, y_tr, X_va, X_te, keys, m=m, n_inner=n_inner)
    X_tr, X_va, X_te = (pd.concat([X.drop(columns=keys), E], axis=1)
                        for X, E in ((X_tr, E_tr), (X_va, E_va), (X_te, E_te)))
    return X_tr, X_va, X_te, keys


def lgbm_vte(params: dict | None = None, early_stopping: int = 300, m: float = 20.0, n_inner: int = 5):
    """H14: fold-internes TE aller Wert-/Ziffern-Schlüssel (String-Spalten mit Suffix `__key`, z. B. aus v6),
    Schlüssel danach entfernen, dann LightGBM mit flachen Bäumen (Defaults: 4 Leaves, lr 0.03, colsample 0.5)."""
    import lightgbm as lgb

    p = dict(objective="binary", learning_rate=0.03, num_leaves=4, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
             n_estimators=20000, random_state=SEED, verbose=-1)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        X_tr, X_va, X_te, keys = vte_frames(X_tr, y_tr, X_va, X_te, m=m, n_inner=n_inner)
        mdl = lgb.LGBMClassifier(**p)
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        print(f"  keys {len(keys)} | features {X_tr.shape[1]} | best_iter {mdl.best_iteration_}", flush=True)
        return mdl.predict_proba(X_va)[:, 1], mdl.predict_proba(X_te)[:, 1]

    return fit_predict


MODELS["lgbm_vte"] = lgbm_vte


def xgb_vte(params: dict | None = None, early_stopping: int = 300, m: float = 20.0, n_inner: int = 5):
    """H15a: gleicher fold-interner TE-Schritt wie lgbm_vte (`vte_frames`), dann XGBoost mit Tiefe 2
    (hist, Kategorien nativ per enable_categorical wie in `xgb`)."""
    import xgboost as xgb_

    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.04, max_depth=2,
             subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0, n_estimators=20000,
             tree_method="hist", enable_categorical=True, random_state=SEED,
             early_stopping_rounds=early_stopping)
    p.update(params or {})

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        X_tr, X_va, X_te, keys = vte_frames(X_tr, y_tr, X_va, X_te, m=m, n_inner=n_inner)
        mdl = xgb_.XGBClassifier(**p)
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        print(f"  keys {len(keys)} | features {X_tr.shape[1]} | best_iter {mdl.best_iteration}", flush=True)
        return mdl.predict_proba(X_va)[:, 1], mdl.predict_proba(X_te)[:, 1]

    return fit_predict


MODELS["xgb_vte"] = xgb_vte


def glm_vte(params: dict | None = None, m: float = 20.0, n_inner: int = 5, clip: float = 1e-4):
    """H15b: GLM = LogisticRegression auf logit(TE) der `__key`-Spalten (TE wie lgbm_vte, geclippt auf
    [clip, 1-clip]) + BOOL_COLS + One-Hot(CAT_COLS); StandardScaler/OneHot fold-intern in der Pipeline.
    Rohe Zahlen und Cents gehen nicht ein (stecken als exakter Wert/Ziffer im TE)."""
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from .config import BOOL_COLS

    lr_p = {"C": 1.0, "max_iter": 3000, **(params or {})}

    def fit_predict(X_tr, y_tr, X_va, y_va, X_te):
        X_tr, X_va, X_te, keys = vte_frames(X_tr, y_tr, X_va, X_te, m=m, n_inner=n_inner)
        te = [f"{c}_te" for c in keys]
        cats = [c for c in CAT_COLS if c in X_tr.columns]

        def prep(X):
            Z = np.clip(X[te].to_numpy(np.float64), clip, 1 - clip)
            out = pd.DataFrame(np.log(Z / (1 - Z)), columns=[f"{c}_logit" for c in te], index=X.index)
            return pd.concat([out, X[BOOL_COLS].astype(np.float64), X[cats]], axis=1)

        P_tr, P_va, P_te = prep(X_tr), prep(X_va), prep(X_te)
        nums = [c for c in P_tr.columns if c not in cats]
        pre = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cats),
                                      (StandardScaler(), nums))
        mdl = make_pipeline(pre, LogisticRegression(**lr_p))
        mdl.fit(P_tr, y_tr)
        print(f"  keys {len(keys)} | features {P_tr.shape[1]} | C {lr_p['C']} | "
              f"n_iter {mdl[-1].n_iter_[0]}", flush=True)
        return mdl.predict_proba(P_va)[:, 1], mdl.predict_proba(P_te)[:, 1]

    return fit_predict


MODELS["glm_vte"] = glm_vte
