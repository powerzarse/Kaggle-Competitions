"""Experiment 15: Familien-Spezialist nur auf train.csv (statt Open Pool) als Apex-Evidenz, Vorbereitung für V20.

Frage für den LB: V15–V17 (Spezialist auf dem Open Pool) haben geschadet, die auf train.csv trainierte
Geschwister-LR von Apex hilft. Liegt das an der Trainingsquelle (Test näher an train.csv als am Pool) oder am
Eingriff selbst (der echte Stack ist innerhalb der Familien stärker als unser Ersatz)? V20 = derselbe Eingriff
wie V15, nur mit einem Spezialisten, der wie die Geschwister-LR zur Laufzeit auf train.csv gefittet wird.

Hier (lokal, OOF über die Prompt-Folds):
1. Wie gut trennt ein TF-IDF + LR-Spezialist mit ~150 Texten je Klasse die Geschwister? (Raster über C/Merkmale)
2. Wie stark greift er in der V6-Kette ein (geänderte Vorhersagen gegen V6), verglichen mit dem Pool-Spezialisten
   bei den Einstellungen von V15/V16? Damit wird die Stärke von V20 so gewählt, dass der LB-Vergleich trägt.
   Das Δ der Simulation wird nur mitgeschrieben; für Nicht-GLM-Entscheidungen ist es zweimal widerlegt.

Aufruf (aus Name_That_LLM_2026/): .venv/bin/python experiments/15_train_spec/oof_train_spec.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import simulate as s9
from src.cv import get_folds
from src.data import load_train

OUT = ROOT / "models" / "15_train_spec"
FAMS = ("claude", "gpt", "gemini")
CONFIGS = {   # Name -> (Wort-Kwargs oder None, Zeichen-Kwargs oder None, C)
    "wort+zeichen C1": (dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), 1.0),
    "wort+zeichen C4": (dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), 4.0),
    "wort+zeichen C16": (dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), 16.0),
    "zeichen C4": (None, dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), 4.0),
    "zeichen C16": (None, dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True), 16.0),
}


def fit_family(texts, target, word, char, C):
    vecs = [TfidfVectorizer(**kw) for kw in (word, char) if kw is not None]
    X = hstack([v.fit_transform(texts) for v in vecs]).tocsr()
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, target)
    return vecs, clf


def predict_family(model, texts):
    vecs, clf = model
    return clf.predict_proba(hstack([v.transform(texts) for v in vecs]).tocsr()), clf.classes_


def oof_spec(texts, y, folds, labels, families, word, char, C):
    """(n, 18) Evidenz wie im Notebook: innerhalb der Familie Spezialist-Wahrscheinlichkeiten, sonst 1."""
    T = np.asarray(texts, dtype=object)
    L = np.asarray(labels)
    ev = np.ones((len(texts), len(labels)))
    for cols in families[:3]:
        fam = np.isin(y, L[cols])
        for k in np.unique(folds):
            tr, va = (folds != k) & fam, folds == k
            p, classes = predict_family(fit_family(T[tr], y[tr], word, char, C), T[va])
            for j, c in enumerate(classes):
                ev[va, labels.index(c)] = p[:, j]
    return ev


def main():
    t0 = time.time()
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    folds = get_folds(real)
    families = res_mod.family_columns(labels)
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
    stage = L[post.argmax(1)]

    def chain(evid, s):
        final, _ = apex.apply_arrays(zl, evid, labels, families, texts, stage, s, False, 0.0)
        return final.astype(str)

    v6 = chain(ev, 0.10)
    base = balanced_accuracy_score(y, v6)

    def within(e):
        out = []
        for cols in families[:3]:
            m = np.isin(y, L[cols])
            out.append((L[cols][e[m][:, cols].argmax(1)] == y[m]).mean())
        return out

    pool = np.load(ROOT / "models" / "12_family_specialist" / "spec_tfidf_train_final.npy")
    print("Innerhalb der Familie (Claude / GPT / Gemini), train.csv OOF:")
    print(f"  {'Ersatz-Stack':<22}", " / ".join(f"{a:.3f}" for a in within(zl)))
    print(f"  {'Geschwister-LR (V6)':<22}", " / ".join(f"{a:.3f}" for a in within(ev)))
    print(f"  {'Pool-Spezialist (V15)':<22}", " / ".join(f"{a:.3f}" for a in within(pool)), flush=True)
    specs = {}
    for name, (word, char, C) in CONFIGS.items():
        specs[name] = oof_spec(texts, y, folds, labels, families, word, char, C)
        print(f"  {name:<22}", " / ".join(f"{a:.3f}" for a in within(specs[name])),
              f"({time.time() - t0:.0f} s)", flush=True)

    rows = []
    grid = [(1.0, 0.15), (1.0, 0.25), (1.5, 0.20), (1.0, 0.35), (1.0, 0.5), (2.0, 0.25), (2.0, 0.35), (2.0, 0.5)]
    for name, spec in [("Pool-Spezialist (V15)", pool)] + list(specs.items()):
        for t, s in grid:
            f = chain(ev * spec ** t, s)
            rows.append(dict(spezialist=name, t=t, strength=s, geaendert=int((f != v6).sum()),
                             d_sim=100 * (balanced_accuracy_score(y, f) - base)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nV6-Kette (Ersatz-Stack, GLM 20 %), Apex-Evidenz LR x Spezialist^t: geänderte Vorhersagen gegen V6 "
          "(Eingriffsstärke) und Δ der Simulation in Pp. (für Nicht-GLM nicht belastbar):")
    print(df.pivot_table(index="spezialist", columns=["t", "strength"], values="geaendert").to_string())
    print()
    print(df.pivot_table(index="spezialist", columns=["t", "strength"], values="d_sim").round(2).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "oof_train_spec.csv", index=False)
    np.savez(OUT / "oof_specs.npz", **{k.replace(" ", "_").replace("+", "_"): v for k, v in specs.items()})

    word, char, C = CONFIGS["wort+zeichen C4"]
    t1 = time.time()
    T = np.asarray(texts, dtype=object)
    for cols in families[:3]:
        m = np.isin(y, L[cols])
        fit_family(T[m], y[m], word, char, C)
    print(f"\nFit auf ganz train.csv (alle drei Familien, wort+zeichen C4): {time.time() - t1:.1f} s")


if __name__ == "__main__":
    main()
