"""Experiment 14b: Wie robust sind die GLM-Köpfe gegen das Laufrauschen der GGUF-Bewertung?

Zur Testzeit kommen die Likelihood-Merkmale aus einem anderen Lauf als die Trainingsmerkmale (andere Batches,
die MoE-GGUF-Durchläufe sind batchabhängig). Für train gibt es jetzt zwei unabhängige Läufe:
- Lauf A: estopadillas train-Lauf (160 Tokens), Merkmale in ll_train.npz (V6 trainiert darauf, + Pool aus unserem Lauf)
- Lauf B: unser Lauf vom 20.11. (512 Tokens, Schnitt 160/256 nachgerechnet, evaluate.py → models/14_glm_256/train_feats.npz)

Bewertet wird OOF (gleiche Folds) mit Training auf A und Bewertung auf B (wie zur Testzeit) gegen A/A. Für die
256er-Köpfe gibt es nur Lauf B; deren Rauschanfälligkeit wird über den 160er-Vergleich abgeschätzt.

Aufruf (aus Name_That_LLM_2026/, nach evaluate.py):
    .venv/bin/python experiments/14_glm_256/evaluate_crossrun.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))
sys.path.insert(0, str(ROOT / "experiments" / "14_glm_256"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

import simulate as s9
from evaluate import lr, hgb, OUT
from src.cv import get_folds
from src.data import load_train


def fit_eval(make, Xtr_run, Xev_run, y, folds, labels, Xpool=None, ypool=None, wpool=None):
    out = np.zeros((len(y), len(labels)))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if Xpool is None:
            clf = make().fit(Xtr_run[tr], y[tr])
        else:   # nur HGB (wie V6): train-Folds Gewicht 1, Pool mit Gewicht aus ll_train.npz
            clf = make().fit(np.vstack([Xtr_run[tr], Xpool]), np.concatenate([y[tr], ypool]),
                             sample_weight=np.concatenate([np.ones(tr.sum()), wpool]))
        p = clf.predict_proba(Xev_run[va])
        out[va] = np.log(np.clip(p[:, [list(clf.classes_).index(c) for c in labels]], 1e-9, 1))
    return out


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)
    n = len(real)
    with np.load(s9.NPZ, allow_pickle=False) as d:
        XA_all, yA_all, wA_all = d["X"].astype(np.float64), d["y"].astype(str), d["w"].astype(np.float64)
    XA, Xpool, ypool, wpool = XA_all[:n], XA_all[n:], yA_all[n:], wA_all[n:]
    with np.load(OUT / "train_feats.npz", allow_pickle=True) as d:
        assert (d["ids"].astype(str) == real.id.astype(str).to_numpy()).all()
        XB160, XB256 = d["X160"], d["X256"]

    corr = pd.Series([np.corrcoef(XA[:, j], XB160[:, j])[0, 1] for j in range(XA.shape[1])], index=ll_model["features"])
    print("Merkmale mit der geringsten Übereinstimmung zwischen Lauf A und B (160 Tokens):")
    print(corr.sort_values().head(8).round(3).to_string(), "\n")

    with np.load(s9.OUT / "oof_cache.npz") as c:
        ev = c["ev"]
    families = res_mod.family_columns(labels)
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    grp = lambda p: np.where(np.isin(p, s9.GLM), p, "rest")

    def chain(ll, cover=0.20):
        post, _ = s9.glm_stage(zl, ll, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=cover)
        final, _ = apex.apply_arrays(zl, ev, labels, families, texts, L[post.argmax(1)], 0.10, False, 0.0)
        return final.astype(str)

    heads = {
        "V6 HGB-160+Pool  A→A (bisherige Sim)": fit_eval(hgb, XA, XA, y, folds, labels, Xpool, ypool, wpool),
        "V6 HGB-160+Pool  A→B (wie Testzeit)": fit_eval(hgb, XA, XB160, y, folds, labels, Xpool, ypool, wpool),
        "HGB-160 train    A→A": fit_eval(hgb, XA, XA, y, folds, labels),
        "HGB-160 train    A→B": fit_eval(hgb, XA, XB160, y, folds, labels),
        "HGB-160 train    B→B": fit_eval(hgb, XB160, XB160, y, folds, labels),
        "LR-160           A→A": fit_eval(lr, XA, XA, y, folds, labels),
        "LR-160           A→B": fit_eval(lr, XA, XB160, y, folds, labels),
        "LR-160           B→B": fit_eval(lr, XB160, XB160, y, folds, labels),
        "LR-256           B→B": fit_eval(lr, XB256, XB256, y, folds, labels),
        "HGB-256 train    B→B": fit_eval(hgb, XB256, XB256, y, folds, labels),
    }
    heads["Mix 0.5 V6(A→B) + LR-256"] = log_softmax(0.5 * heads["V6 HGB-160+Pool  A→B (wie Testzeit)"] + 0.5 * heads["LR-256           B→B"], 1)
    heads["Mix 0.7 V6(A→B) + 0.3 LR-256"] = log_softmax(0.7 * heads["V6 HGB-160+Pool  A→B (wie Testzeit)"] + 0.3 * heads["LR-256           B→B"], 1)

    ref = chain(heads["V6 HGB-160+Pool  A→B (wie Testzeit)"])
    base = balanced_accuracy_score(y, ref)
    idx = np.array([labels.index(c) for c in y])
    rows = []
    for name, ll in heads.items():
        p = L[ll.argmax(1)]
        row = dict(kopf=name, kopf_bal=balanced_accuracy_score(y, p), nll=-ll[np.arange(n), idx].mean(),
                   kopf_glm3=balanced_accuracy_score(grp(y), grp(p)))
        for cover in (0.20, 0.167):
            f = chain(ll, cover)
            row[f"kette_{cover:.0%}"] = balanced_accuracy_score(y, f)
            row[f"d_{cover:.0%}"] = 100 * (balanced_accuracy_score(y, f) - base)
        m, lo, hi, pp = s9.bootstrap(y, ref, chain(ll, 0.20), real.prompt_id, n=1000)
        row.update(lo20=100 * lo, hi20=100 * hi)
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("Δ in Pp. gegen V6 A→B bei 20 % (= realistische V6-Referenz):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(OUT / "results_crossrun.csv", index=False)


if __name__ == "__main__":
    main()
