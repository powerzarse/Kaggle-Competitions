"""Experiment 12, Schnelltest: Familien-Spezialisten (TF-IDF + linearer Klassifikator) nur auf dem Open Pool.

Frage: Trennt ein Spezialist, der pro Familie (Claude 5, GPT 5, Gemini 4 Klassen) auf allen Pool-Texten der Familie
trainiert ist, die Geschwister besser als der Stack? Gemessen auf train.csv (nicht im Pool, also sauber getrennt):
- Trefferquote innerhalb der Familie (Texte der Familie, argmax nur über die Familie)
- V6-Kette (Ersatz-Stack -> GLM-Stufe 20 % -> Apex), Apex-Evidenz = Spezialist statt (oder mal) Geschwister-LR,
  Stärke 0.10 (wie V6) bis 0.6.

Aufruf (aus Name_That_LLM_2026/): .venv/bin/python experiments/12_family_specialist/quick_tfidf.py
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
from scipy.special import log_softmax, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import simulate as s9
from src.data import load_train

OUT = ROOT / "models" / "12_family_specialist"
FAMS = ("claude", "gpt", "gemini")
CAP = 15000   # höchstens so viele Pool-Texte je Klasse (Laufzeit)


def main():
    t0 = time.time()
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["text", "model"])
    pool = pd.concat([g.sample(min(len(g), CAP), random_state=0) for _, g in pool.groupby("model")])
    families = res_mod.family_columns(labels)

    spec = np.ones((len(real), len(labels)))   # Evidenz: Spezialist-Wahrscheinlichkeit je Familie, sonst 1
    for fam, cols in zip(("claude", "gpt", "gemini", "glm"), families):
        if fam not in FAMS:
            continue
        cl = L[cols]
        P = pool[pool.model.isin(cl)]
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400000, sublinear_tf=True)
        Xp = hstack([vw.fit_transform(P.text), vc.fit_transform(P.text)]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced", solver="saga")
        clf.fit(Xp, P.model)
        Xt = hstack([vw.transform(texts), vc.transform(texts)]).tocsr()
        p = clf.predict_proba(Xt)
        for j, c in enumerate(clf.classes_):
            spec[:, labels.index(c)] = p[:, j]
        m = np.isin(y, cl)
        acc = (L[cols][spec[m][:, cols].argmax(1)] == y[m]).mean()
        print(f"{fam}: {len(P)} Pool-Texte, Spezialist innerhalb der Familie {acc:.3f} ({time.time()-t0:.0f} s)", flush=True)

    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    print("\nInnerhalb der Familie (train.csv, Texte der Familie):")
    for fam, cols in zip(("claude", "gpt", "gemini"), families[:3]):
        m = np.isin(y, L[cols])
        f = lambda e: (L[cols][e[m][:, cols].argmax(1)] == y[m]).mean()
        print(f"  {fam:7s} Stack {f(zl):.3f} | Geschwister-LR {f(ev):.3f} | Spezialist {f(spec):.3f} | "
              f"Stack+Spezialist {f(zl + np.log(np.clip(spec, 1e-9, 1))):.3f}")

    post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
    stage = L[post.argmax(1)]
    rows = []
    for name, evid in (("Geschwister-LR (V6)", ev), ("Spezialist", spec), ("LR x Spezialist", ev * spec)):
        for s in (0.10, 0.2, 0.3, 0.45, 0.6, 1.0):
            final, _ = apex.apply_arrays(zl, evid, labels, families, texts, stage, s, False, 0.0)
            rows.append(dict(evidenz=name, strength=s, bal_acc=balanced_accuracy_score(y, final.astype(str))))
    df = pd.DataFrame(rows)
    ref = df[(df.evidenz == "Geschwister-LR (V6)") & (df.strength == 0.10)].bal_acc.iloc[0]
    df["d_v6"] = 100 * (df.bal_acc - ref)
    print("\nV6-Kette (Ersatz-Stack, GLM 20 %), Apex mit anderer Evidenz:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "spec_tfidf_train.npy", spec)
    df.to_csv(OUT / "quick_tfidf.csv", index=False)


if __name__ == "__main__":
    main()
