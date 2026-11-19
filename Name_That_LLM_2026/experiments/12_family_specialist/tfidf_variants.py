"""Experiment 12: Varianten des TF-IDF-Spezialisten (Datenmenge, Regularisierung), bewertet wie quick_tfidf.py.

Aufruf: .venv/bin/python experiments/12_family_specialist/tfidf_variants.py --cap 0 --C 4 --tag nocap_c4
(--cap 0 = alle Pool-Texte; Ausgabe models/12_family_specialist/spec_{tag}_train.npy und eine Zeile Ergebnis)
"""
import argparse
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
from sklearn.metrics import balanced_accuracy_score as bas

import simulate as s9
from src.data import load_train

OUT = ROOT / "models" / "12_family_specialist"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=15000)
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-iter", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["text", "model"])
    if a.cap > 0:
        pool = pd.concat([g.sample(min(len(g), a.cap), random_state=0) for _, g in pool.groupby("model")])
    fams = res_mod.family_columns(labels)
    spec = np.ones((len(real), len(labels)))
    accs = {}
    for fam, cols in zip(("claude", "gpt", "gemini"), fams[:3]):
        P = pool[pool.model.isin(L[cols])]
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400000, sublinear_tf=True)
        X = hstack([vw.fit_transform(P.text), vc.fit_transform(P.text)]).tocsr()
        clf = LogisticRegression(C=a.C, max_iter=a.max_iter, class_weight="balanced", solver="saga", random_state=0).fit(X, P.model)
        p = clf.predict_proba(hstack([vw.transform(texts), vc.transform(texts)]).tocsr())
        for j, c in enumerate(clf.classes_):
            spec[:, labels.index(c)] = p[:, j]
        m = np.isin(y, L[cols])
        accs[fam] = (L[cols][spec[m][:, cols].argmax(1)] == y[m]).mean()
        print(f"{a.tag} {fam}: {len(P)} Texte, innerhalb {accs[fam]:.3f} ({time.time() - t0:.0f} s)", flush=True)
    np.save(OUT / f"spec_{a.tag}_train.npy", spec)
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
    stage = L[post.argmax(1)]
    ref, _ = apex.apply_arrays(zl, ev, labels, fams, texts, stage, 0.10, False, 0.0)
    ref = ref.astype(str)
    res = {}
    for t in (1.0, 1.5):
        for s in (0.15, 0.2, 0.25, 0.3):
            f, _ = apex.apply_arrays(zl, ev * spec ** t, labels, fams, texts, stage, s, False, 0.0)
            res[(t, s)] = round(100 * (bas(y, f.astype(str)) - bas(y, ref)), 2)
    print(f"ERGEBNIS {a.tag}: innerhalb {accs} | Kette {res}", flush=True)


if __name__ == "__main__":
    main()
