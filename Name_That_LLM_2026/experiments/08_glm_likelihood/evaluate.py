"""Experiment 08: GLM-Likelihood-Klassifikator lokal bewerten (ohne Kaggle-GPU).

Baut die GLM-Stufe des öffentlichen 0.694-Notebooks (v1src/v1_stage.py) auf train.csv nach:
- Klassifikator: multinomiale LogReg auf 54 Likelihood-Merkmalen (Flash + GLM-5.3 + 6 Differenzen), 160 Tokens.
  OOF über dieselben GroupKFold-Folds wie überall im Projekt (src/cv.py).
- Stack-Ersatz: unser eigenes Ensemble (BERT 0.4 / SVM 0.4 / LoRA 0.2, OOF ~0.60). Der echte Stack ist stärker,
  die Zahlen sind also ein Vergleich zwischen Klassifikator-Varianten, kein LB-Schätzer.
- Stufe: Auswahl nach GLM-Masse (bzw. "uncertain"), Temperatur auf entropy_ref, argmax(zl + W/T' * ll).

Optional: zusätzliche Pool-Merkmale (Parquet je Modell-Tag wie train) mit Gewicht --pool-weight.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/08_glm_likelihood/evaluate.py
    .venv/bin/python experiments/08_glm_likelihood/evaluate.py --pool models/08_glm_likelihood/pool_feats_L160.parquet
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent / "v1src"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.cv import get_folds
from src.data import load_train
from v1_stage import match_temperature

MODEL = json.loads((Path(__file__).parent / "v1src" / "ll_model.json").read_text())
FEATS = MODEL["features"]
GLM = ["glm-5.3", "glm-5.3-flash"]


def stack_proxy(real, classes):
    m = ROOT / "models" / "07_kaggle"
    load = lambda p: pd.read_parquet(p).set_index("id").loc[real["id"], classes].values
    P = (0.4 * load(m / "bert" / "oof_stage2_ep2.parquet")
         + 0.4 * softmax(load(m / "svm" / "svm_oof_decision.parquet") * 5, 1)
         + 0.2 * load(m / "lora" / "oof_stage2.parquet"))
    return np.log(np.clip(P, 1e-9, 1))


def fit_lr(X, y, w, C):
    sc = StandardScaler().fit(X, sample_weight=w)
    lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def oof_ll(Xtr, ytr, folds, classes, C, Xpool=None, ypool=None, pool_w=1.0):
    out = np.zeros((len(ytr), len(classes)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        X, y, w = Xtr[tr], ytr[tr], np.ones(tr.sum())
        if Xpool is not None:
            # Pool ist bewusst GLM-lastig ausgewählt: Klassen auf gleiches Gesamtgewicht bringen,
            # sonst verschiebt sich der Prior des Klassifikators Richtung GLM.
            cnt = pd.Series(ypool).value_counts()
            bal = (cnt.mean() / cnt.reindex(ypool).to_numpy())
            X = np.vstack([X, Xpool]); y = np.concatenate([y, ypool]); w = np.concatenate([w, pool_w * bal])
        sc, lr = fit_lr(X, y, w, C)
        p = lr.predict_log_proba(sc.transform(Xtr[va]))
        out[va] = p[:, [list(lr.classes_).index(c) for c in classes]]
    return out


def stage(zl, ll, classes, y, mode="mass", sel_cls=GLM, cover=0.20, W=1.0):
    idx = [classes.index(c) for c in sel_cls]
    mass = np.exp(zl[:, idx]).sum(1)
    score = mass * (1 - np.exp(zl).max(1)) if mode == "uncertain" else mass
    n_sel = int(cover * len(zl))
    sel = np.sort(np.argsort(-score, kind="stable")[:n_sel])
    tp = match_temperature(zl, MODEL["entropy_ref"])
    before = zl.argmax(1); after = before.copy()
    after[sel] = (zl[sel] + W / tp * ll[sel]).argmax(1)
    C = np.asarray(classes)
    return balanced_accuracy_score(y, C[before]), balanced_accuracy_score(y, C[after])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="Parquet mit Pool-Merkmalen (Spalten wie train + 'model')")
    ap.add_argument("--pool-weight", type=float, nargs="+", default=[1.0])
    ap.add_argument("--C", type=float, nargs="+", default=[MODEL["C"]])
    a = ap.parse_args()

    real = load_train()
    classes = list(MODEL["classes"])
    y = real["model"].values
    folds = get_folds(real)
    F = pd.read_parquet(ROOT / "models" / "08_glm_likelihood" / "train_v3" / "train_feats_L160.parquet")
    X = F.loc[real["id"].astype(str), FEATS].to_numpy(np.float64)
    zl = stack_proxy(real, classes)
    glm_mask = np.isin(y, GLM)

    pool = pd.read_parquet(a.pool) if a.pool else None
    configs = [(C, None) for C in a.C] + ([(C, w) for C in a.C for w in a.pool_weight] if pool is not None else [])
    print(f"{'Variante':<28} {'LL bal.acc':>10} {'LL logloss':>10} {'GLM-Recall':>10} | Stack vorher -> mass20 / uncertain30")
    for C, w in configs:
        ll = oof_ll(X, y, folds, classes, C,
                    *((pool[FEATS].to_numpy(np.float64), pool["model"].values, w) if w is not None else ()))
        pred = np.asarray(classes)[ll.argmax(1)]
        b0, b1 = stage(zl, ll, classes, y, "mass", GLM, 0.20)
        _, b2 = stage(zl, ll, classes, y, "uncertain", GLM + ["claude-fable-5"], 0.30)
        name = f"C={C}" + (f" + Pool w={w}" if w is not None else " (nur train)")
        print(f"{name:<28} {balanced_accuracy_score(y, pred):>10.4f} {log_loss(y, np.exp(ll), labels=classes):>10.4f} "
              f"{(pred[glm_mask] == y[glm_mask]).mean():>10.4f} | {b0:.4f} -> {b1:.4f} / {b2:.4f}")


if __name__ == "__main__":
    main()
