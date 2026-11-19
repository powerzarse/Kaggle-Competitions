"""Experiment 12: Familien-Spezialisten (TF-IDF + logistische Regression, nur Open Pool) fitten und exportieren.

Gleiche Einstellungen wie quick_tfidf.py (dort simuliert: Apex-Evidenz LR x Spezialist, Stärke 0.25 -> +0,73 Pp. in
der V6-Kette auf dem Ersatz-Stack), nur mit festem random_state. Export ohne Pickle (Versionen im Abgabe-Image):
je Familie `{fam}_vocab_word.json`, `{fam}_vocab_char.json` (Liste der Terme in Spaltenreihenfolge) und
`{fam}.npz` (idf_word, idf_char, coef, intercept, classes). `spec_evidence()` unten ist dieselbe Funktion wie im
Notebook; sie wird gegen die sklearn-Modelle geprüft.

Aufruf: .venv/bin/python experiments/12_family_specialist/export_spec.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_train

OUT = ROOT / "models" / "12_family_specialist" / "spec"
FAMS = {"claude": "claude-", "gpt": "gpt-", "gemini": "gemini-"}
CAP = 15000
WORD = dict(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400000, sublinear_tf=True)


def spec_evidence(texts, spec_dir, labels):
    """(n, 18) Evidenz: innerhalb jeder Familie die Wahrscheinlichkeiten des Spezialisten, sonst 1 (wie im Notebook)."""
    from scipy.special import softmax
    ev = np.ones((len(texts), len(labels)))
    for fam in ("claude", "gpt", "gemini"):
        z = np.load(Path(spec_dir) / f"{fam}.npz", allow_pickle=False)
        mats = []
        for kind, kw in (("word", WORD), ("char", CHAR)):
            terms = json.loads((Path(spec_dir) / f"{fam}_vocab_{kind}.json").read_text())
            kw = {k: v for k, v in kw.items() if k not in ("min_df", "max_features")}
            v = TfidfVectorizer(vocabulary={t: i for i, t in enumerate(terms)}, **kw)
            v.idf_ = z[f"idf_{kind}"]
            mats.append(v.transform(texts))
        X = hstack(mats).tocsr()
        logits = X @ z["coef"].T + z["intercept"]
        p = softmax(logits, axis=1)
        for j, c in enumerate(z["classes"].astype(str)):
            ev[:, labels.index(c)] = p[:, j]
    return ev


def main():
    t0 = time.time()
    real = load_train()
    labels = sorted(real.model.unique())
    texts = real.text.fillna("").astype(str).tolist()
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["text", "model"])
    pool = pd.concat([g.sample(min(len(g), CAP), random_state=0) for _, g in pool.groupby("model")])
    OUT.mkdir(parents=True, exist_ok=True)
    ref = np.ones((len(real), len(labels)))
    for fam, pre in FAMS.items():
        P = pool[pool.model.str.startswith(pre)]
        vw, vc = TfidfVectorizer(**WORD), TfidfVectorizer(**CHAR)
        Xp = hstack([vw.fit_transform(P.text), vc.fit_transform(P.text)]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced", solver="saga", random_state=0)
        clf.fit(Xp, P.model)
        for kind, v in (("word", vw), ("char", vc)):
            terms = [None] * len(v.vocabulary_)
            for t, i in v.vocabulary_.items():
                terms[i] = t
            (OUT / f"{fam}_vocab_{kind}.json").write_text(json.dumps(terms, ensure_ascii=False))
        np.savez_compressed(OUT / f"{fam}.npz", idf_word=vw.idf_, idf_char=vc.idf_,
                            coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
                            classes=np.asarray(clf.classes_, dtype=str))
        p = clf.predict_proba(hstack([vw.transform(texts), vc.transform(texts)]).tocsr())
        for j, c in enumerate(clf.classes_):
            ref[:, labels.index(c)] = p[:, j]
        print(f"{fam}: {len(P)} Texte, {Xp.shape[1]} Merkmale ({time.time() - t0:.0f} s)", flush=True)
    ev = spec_evidence(texts, OUT, labels)
    err = np.abs(ev - ref).max()
    print(f"Nachbau gegen sklearn: max |Δp| = {err:.2e}")
    assert err < 1e-4, err
    np.save(OUT.parent / "spec_tfidf_train_final.npy", ev)
    old = np.load(OUT.parent / "spec_tfidf_train.npy")
    print(f"gegen den simulierten Spezialisten (saga ohne festen Seed): max |Δp| = {np.abs(old - ev).max():.3f}, "
          f"argmax gleich {((old.argmax(1) == ev.argmax(1)) | (old.max(1) == 1)).mean():.4f}")
    print("Dateien:", sorted(p.name for p in OUT.iterdir()), f"{sum(p.stat().st_size for p in OUT.iterdir()) / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
