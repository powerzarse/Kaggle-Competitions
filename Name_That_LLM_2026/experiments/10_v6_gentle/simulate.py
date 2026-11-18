"""Experiment 10: weniger aggressive Varianten der GLM-Stufe in der V6-Kette, lokal auf train.csv.

Kette wie V6 (kaggle/public_701/v6_hgb): Stack-Logits -> GLM-Stufe (Top x % nach GLM-Masse, zl + W/T' * ll)
-> Apex auf den Stack-Logits (Residual + harte Typografie; GLM-Vorhersagen der Stufe bleiben, jede andere wird
durch die verfeinerte Stack-Vorhersage ersetzt). In V6 wirkt die Stufe also nur über die Frage „welche Texte
werden GLM“. Genau das soll diese Simulation bewerten, nicht die Nicht-GLM-Klassen (siehe Experiment 09: dort
hat der Ersatz-Stack den LB falsch vorhergesagt).

Varianten (alle auf derselben Kette, Abdeckung 20 % ≈ V6 und 30 % = Höchstwert von V6):
- W-Raster (V12: W 0.7). Eine Temperatur auf dem Klassifikator ist für die argmax dasselbe wie W/T.
- „GLM gegen den Rest“ (V13): der Klassifikator liefert nur p(GLM-5.3), p(Flash), p(Rest); p(Rest) wird
  gleichmäßig auf die 16 Nicht-GLM-Klassen verteilt (r2) bzw. jede bekommt p(Rest) (r1, konservativer).
- Kalibriert (V14): HGB in CalibratedClassifierCV (sigmoid / isotonisch, innere GroupKFold), Pool-Gewicht 0.25.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/10_v6_gentle/simulate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

import simulate as s9   # Experiment 09: Module aus dem Notebook, Stack-Ersatz, OOF-Klassifikator, Bootstrap
from src.cv import get_folds
from src.data import load_train

OUT = ROOT / "models" / "10_v6_gentle"
GLM = s9.GLM


def oof_variant(X, y, w, n_train, folds, groups, labels, pool_scale=1.0, calib=None):
    """OOF-Log-Wahrscheinlichkeiten des Klassifikators auf train.csv (wie s9.oof_ll, mit Optionen)."""
    Xt, yt = X[:n_train], y[:n_train]
    Xp, yp, wp = X[n_train:], y[n_train:], w[n_train:] * pool_scale
    out = np.zeros((n_train, len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        Xf, yf = np.vstack([Xt[tr], Xp]), np.concatenate([yt[tr], yp])
        wf = np.concatenate([np.ones(tr.sum()), wp])
        hgb = HistGradientBoostingClassifier(random_state=0, **s9.HGB)
        if calib:
            gf = np.concatenate([groups[tr], [f"pool{i}" for i in range(len(yp))]])
            cv = list(GroupKFold(5).split(Xf, yf, gf))
            clf = CalibratedClassifierCV(hgb, method=calib, cv=cv).fit(Xf, yf, sample_weight=wf)
        else:
            clf = hgb.fit(Xf, yf, sample_weight=wf)
        p = clf.predict_proba(Xt[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
        print(f"  fold {k} fertig ({'pool x' + str(pool_scale) if not calib else calib})", flush=True)
    return out


def glm_vs_rest(ll, labels, split="uniform"):
    """Klassifikator auf {GLM-5.3, Flash, Rest} reduzieren; Verteilung im Rest bestimmt allein der Stack."""
    g = np.isin(labels, GLM)
    p = np.exp(ll)
    rest = p[:, ~g].sum(1, keepdims=True)
    q = p.copy()
    q[:, ~g] = rest / (~g).sum() if split == "uniform" else rest
    return np.log(np.clip(q, 1e-9, None))


def temperature(ll, y, labels):
    idx = np.array([labels.index(c) for c in y])
    nll = lambda T: -log_softmax(ll / T, 1)[np.arange(len(y)), idx].mean()
    r = minimize_scalar(nll, bounds=(0.3, 5), method="bounded")
    return r.x, nll(1.0), r.fun


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)
    groups = real.prompt_id.astype(str).to_numpy()

    with np.load(s9.NPZ, allow_pickle=False) as d:
        X, yy, w = d["X"].astype(np.float64), d["y"].astype(str), d["w"]
    assert (yy[:len(real)] == y).all()
    with np.load(s9.OUT / "oof_cache.npz") as c:   # Experiment 09: HGB + Pool 0.5 (= V6) und Residual-Evidenz
        ll_v6, ev = c["ll"], c["ev"]
    families = res_mod.family_columns(labels)

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "oof_variants.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    jobs = {"pool025": dict(pool_scale=0.5), "sigmoid": dict(calib="sigmoid"), "isotonic": dict(calib="isotonic")}
    for name, kw in jobs.items():
        if name not in have:
            print(f"OOF {name} ...", flush=True)
            have[name] = oof_variant(X, yy, w, len(real), folds, groups, labels, **kw)
            np.savez(cache, **have)

    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    T, n1, nT = temperature(ll_v6, y, labels)
    print(f"Temperatur des V6-Klassifikators auf OOF: T = {T:.2f} (NLL {n1:.4f} -> {nT:.4f}); "
          f"für die argmax gleichwertig zu W = {1 / T:.2f}\n")

    def chain(ll, W=1.0, cover=0.20):
        post, sel = s9.glm_stage(zl, ll, labels, ll_model["entropy_ref"], v1.match_temperature, W=W, cover=cover)
        stage = L[post.argmax(1)]
        final, _ = apex.apply_arrays(zl, ev, labels, families, texts, stage, 0.10, False, 0.0)
        return stage, final.astype(str)

    lls = {"V6 (HGB, Pool 0.5)": ll_v6,
           "GLM gegen Rest, gleichmäßig (r2)": glm_vs_rest(ll_v6, labels, "uniform"),
           "GLM gegen Rest, je p(Rest) (r1)": glm_vs_rest(ll_v6, labels, "full"),
           "Pool-Gewicht 0.25": have["pool025"],
           "kalibriert sigmoid": have["sigmoid"],
           "kalibriert isotonisch": have["isotonic"]}
    grid = [("V6 (HGB, Pool 0.5)", W) for W in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5)]
    grid += [(n, W) for n in list(lls)[1:] for W in (0.7, 1.0)]

    glm_true = np.isin(y, GLM)
    group = lambda p: np.where(np.isin(p, GLM), p, "rest")
    rows = []
    finals = {}
    for cover in (0.20, 0.30):
        _, ref = chain(ll_v6, 1.0, cover)
        for name, W in grid:
            stage, final = chain(lls[name], W, cover)
            finals[(name, W, cover)] = final
            rows.append(dict(
                cover=f"{cover:.0%}", variante=name, W=W,
                bal_acc=balanced_accuracy_score(y, final),
                d_v6=100 * (balanced_accuracy_score(y, final) - balanced_accuracy_score(y, ref)),
                glm_bal3=balanced_accuracy_score(group(y), group(final)),
                rec_glm53=(final[y == "glm-5.3"] == "glm-5.3").mean(),
                rec_flash=(final[y == "glm-5.3-flash"] == "glm-5.3-flash").mean(),
                fp_glm=np.isin(final[~glm_true], GLM).mean(),
                n_glm=int(np.isin(final, GLM).sum()),
                diff_v6=int((final != ref).sum())))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(OUT / "results.csv", index=False)

    print("\nBootstrap über Prompts gegen V6 (W 1.0), Δ bal. acc. in Pp. [90 %], P(>0):")
    for cover in (0.20, 0.30):
        ref = finals[("V6 (HGB, Pool 0.5)", 1.0, cover)]
        for key in [("V6 (HGB, Pool 0.5)", 0.7), ("GLM gegen Rest, gleichmäßig (r2)", 1.0),
                    ("Pool-Gewicht 0.25", 1.0), ("kalibriert sigmoid", 1.0), ("kalibriert isotonisch", 1.0)]:
            m, lo, hi, pp = s9.bootstrap(y, ref, finals[key + (cover,)], real.prompt_id, n=1000)
            print(f"  {cover:.0%}  {key[0]:<34} W {key[1]:<4} {100 * m:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]  {pp:.2f}")


if __name__ == "__main__":
    main()
