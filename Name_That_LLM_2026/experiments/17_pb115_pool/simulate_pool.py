"""Experiment 17: Hilft der Open Pool auch dem PB-115-Kopf des 0.724-Notebooks?

Der PB-115-Kopf (lumibee v69, unser V21) wird nur auf den 2.699 train-Texten trainiert: 54 Zusammenfassungen +
61 token-gleiche Merkmale (GLM-5.3 gegen Flash), StandardScaler, 0.55 LR (C 0.025) + 0.45 Mittel aus 3 LGB.
In der Stufe: Hybrid-Mischung (1 − α) · W/T18 · ll18 + α · w3/T3 · marginal-log-odds, α 0.35, W 1.5,
Abdeckung bis 40 % nach GLM-Masse des Stacks.

Bis die Prefix-B-Scores für den Pool vorliegen (Kernel ntl-glm-scores-poolpb160), prüfen wir die Frage auf unseren
eigenen Läufen mit dem alten Prefix (Reasoning Effort: Max, leerer Nutzer-Turn): train 2.699 + Pool 4.734 Texte
(Experiment 14), Schnitt 160 Tokens. Merkmale mit genau den Funktionen des 0.724-Notebooks (`features.one`,
`v1_token_features`), Kopf mit genau dessen Rezept; OOF über die Prompt-Folds.

Kette (nur GLM-Entscheidung, der Simulation dafür vertrauen wir): Ersatz-Stack → Auswahl nach GLM-Masse
(30 % / 40 %) → Hybrid-Mischung → argmax. Ohne Apex/Sibling Odds/Fable-Guard des echten Notebooks.

Aufruf (aus Name_That_LLM_2026/, LightGBM braucht die libomp aus sklearn):
    DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs \
        .venv/bin/python experiments/17_pb115_pool/simulate_pool.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))
sys.path.insert(0, str(ROOT / "experiments" / "14_glm_256"))
sys.path.insert(0, str(HERE / "v1src"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import features                      # identisch mit experiments/08_glm_likelihood/v1src/features.py
import prefix_variant_stage as pvs   # build_54_matrix, marginal_logodds, compute_hybrid_delta
import simulate as s9
import v1_token_features as vtf
from src.cv import get_folds
from src.data import load_train

spec16 = importlib.util.spec_from_file_location("ev16", ROOT / "experiments" / "16_paired_tokens" / "evaluate.py")
ev16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(ev16)

TRAIN = ROOT / "models" / "14_glm_256" / "train512" / "out"
POOL = ROOT / "models" / "14_glm_256" / "pool512" / "out"
OUT = ROOT / "models" / "17_pb115_pool"
CUT = 160
SPEC = json.loads((HERE / "v1src" / "spec.json").read_text())
LL = json.loads((HERE / "v1src" / "ll_model.json").read_text())
W, ALPHA = LL["W"], 0.35


def feats115(pairs, ids, n_chars):
    base = {"flash": {}, "glm53": {}}
    extra = {}
    for i in ids:
        sf, sg = pairs[i][0][:CUT], pairs[i][1][:CUT]
        base["flash"][i] = features.one(sf, n_chars[i])
        base["glm53"][i] = features.one(sg, n_chars[i])
        row = {}
        row.update(vtf._single_extra(sf, "flash"))
        row.update(vtf._single_extra(sg, "glm53"))
        row.update(vtf._paired_token_features(sf, sg))
        extra[i] = row
    tab = {t: pd.DataFrame(base[t]).T for t in base}
    X54 = pvs.build_54_matrix(tab, list(ids), SPEC["features_54"], SPEC["diff_features"])
    X61 = pd.DataFrame(extra).T.loc[list(ids)][SPEC["features_61"]].to_numpy(np.float64)
    X = np.hstack([X54, X61])
    assert np.isfinite(X).all() and X.shape == (len(ids), 115)
    return X


def pb115_fit_predict(X_tr, y_tr, w_tr, X_te, labels):
    """Rezept aus fit_and_predict_pb115, mit Gewichten für Pool-Zeilen."""
    import lightgbm as lgb
    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)
    lr = LogisticRegression(C=0.025, max_iter=1000).fit(A, y_tr, sample_weight=w_tr)
    p_lr = lr.predict_proba(B)[:, [list(lr.classes_).index(c) for c in labels]]
    p_lgb = []
    for sd in (42, 1042, 2042):
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(labels), n_estimators=220, learning_rate=0.03,
                                 num_leaves=15, min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
                                 reg_alpha=1.0, reg_lambda=5.0, random_state=sd, verbose=-1, n_jobs=4)
        clf.fit(A, y_tr, sample_weight=w_tr)
        p_lgb.append(clf.predict_proba(B)[:, [list(clf.classes_).index(c) for c in labels]])
    p = np.clip(0.55 * p_lr + 0.45 * np.mean(p_lgb, axis=0), 1e-8, 1.0)
    return np.log(p)


def pool_weights(yp, w, balanced):
    """Gewicht je Pool-Zeile; balanced: jede Klasse bekommt im Pool dieselbe Gesamtmasse (Mittel bleibt w)."""
    if not balanced:
        return np.full(len(yp), w)
    cls, cnt = np.unique(yp, return_counts=True)
    per = dict(zip(cls, len(yp) / (len(cls) * cnt)))
    return w * np.array([per[c] for c in yp])


def oof(X, y, folds, labels, Xp=None, yp=None, pool_w=0.0, balanced=False):
    out = np.zeros((len(y), len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if Xp is None or pool_w <= 0:
            out[va] = pb115_fit_predict(X[tr], y[tr], np.ones(tr.sum()), X[va], labels)
        else:
            out[va] = pb115_fit_predict(np.vstack([X[tr], Xp]), np.concatenate([y[tr], yp]),
                                        np.concatenate([np.ones(tr.sum()), pool_weights(yp, pool_w, balanced)]),
                                        X[va], labels)
        print(f"  Fold {k} fertig", flush=True)
    return out


def chain(zl, ll, labels, cover, apex_like=False):
    """apex_like: wie die Apex-Zelle von v69 – Nicht-GLM-Vorhersagen fallen auf den Stack zurück (dort: refine des
    Stacks, hier: argmax des Ersatz-Stacks), außer Fable-Return-geschützte (Stack GLM → Stufe Nicht-GLM)."""
    mass = np.exp(zl[:, [labels.index(c) for c in s9.GLM]]).sum(1)
    sel = np.sort(np.argsort(-mass, kind="stable")[:int(cover * len(zl))])
    delta, _ = pvs.compute_hybrid_delta(zl, ll[sel], labels, W, LL["entropy_ref"], hybrid_alpha=ALPHA)
    post = zl.copy()
    post[sel] += delta
    pred = post.argmax(1)
    if not apex_like:
        return pred
    g = [labels.index(c) for c in s9.GLM]
    keep = np.isin(pred, g) | (np.isin(zl.argmax(1), g) & ~np.isin(pred, g))
    return np.where(keep, pred, zl.argmax(1))


def main():
    real = load_train()
    real["id"] = real.id.astype(str)
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    folds = get_folds(real)

    tp = ev16.load_pairs(TRAIN)
    X = feats115(tp, real.id.tolist(), ev16.n_chars_via_csv(real))
    pp = ev16.load_pairs(POOL)
    lab = pd.read_csv(POOL / "labels.csv", dtype={"id": str}).set_index("id").model
    pids = [i for i in lab.index if i in pp]
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["id", "text"])
    pool = pool[pool.id.astype(str).isin(set(pids))].astype({"id": str})
    Xp = feats115(pp, pids, ev16.n_chars_via_csv(pool))
    yp = lab.loc[pids].to_numpy().astype(str)
    print(f"train {len(y)}, Pool {len(pids)} Texte, 115 Merkmale", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "oof_heads.npz"
    heads = {}
    if cache.exists():
        with np.load(cache) as c:
            heads = {k: c[k] for k in c.files}
    for name, w, bal in (("nur train (wie 0.724)", 0.0, False), ("train + Pool w0.5", 0.5, False),
                         ("train + Pool w1.0", 1.0, False), ("train + Pool ausgeglichen w0.5", 0.5, True),
                         ("train + Pool ausgeglichen w1.0", 1.0, True)):
        if name not in heads:
            print(name, flush=True)
            heads[name] = oof(X, y, folds, labels, Xp, yp, w, bal)
            np.savez(cache, **heads)

    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    grp = lambda p: np.where(np.isin(p, s9.GLM), p, "rest")
    idx = np.array([labels.index(c) for c in y])
    glm = np.isin(y, s9.GLM)
    rows = []
    for cover, apex_like in ((0.40, False), (0.30, True), (0.40, True)):
        ref = L[chain(zl, heads["nur train (wie 0.724)"], labels, cover, apex_like)]
        for name, ll in heads.items():
            f = L[chain(zl, ll, labels, cover, apex_like)]
            d, lo, hi, pg = s9.bootstrap(y, ref, f, real.prompt_id, n=1000)
            rec = lambda c: float((f[y == c] == c).mean())
            rows.append(dict(abdeckung=cover, apex=apex_like, kopf=name,
                             kopf_bal=balanced_accuracy_score(y, L[ll.argmax(1)]),
                             kopf_glm3=balanced_accuracy_score(grp(y), grp(L[ll.argmax(1)])),
                             nll_glm=-ll[glm, idx[glm]].mean(),
                             kette=balanced_accuracy_score(y, f), d_pp=100 * d, lo=100 * lo, hi=100 * hi, p_gt0=pg,
                             rec_glm53=rec("glm-5.3"), rec_flash=rec("glm-5.3-flash"), rec_fable=rec("claude-fable-5"),
                             geaendert=int((f != ref).sum())))
        print(f"Abdeckung {cover:.0%}: Stack allein {balanced_accuracy_score(y, L[zl.argmax(1)]):.4f}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nΔ in Pp. gegen 'nur train' bei gleicher Abdeckung (Bootstrap 90 % über Prompts):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(OUT / "results.csv", index=False)


if __name__ == "__main__":
    main()
