"""Experiment 18, Vortest: Helfen host-ähnlich erzeugte Texte (markbarney) den TF-IDF/SVC-Lesern des 0.727-Kerns?

Der vorab trainierte Kern des 0.727-Notebooks (`ntl666-prefit`, Rezept areebbb/ntl-we-back-0-666 V14) besteht aus
svc4 (LinearSVC, 4 Ansichten; train w1 + nlm-matched-clean w0.5 + Open Pool v13 w0.35, Blendgewicht 12),
tfidf (LR, 2 Ansichten; train + matched-clean, 0.25) und distort (0.10). Hier ohne Pool (schnell): Leser auf
train-Folds + matched-clean (w0.5) ± markbarney 5 Modelle × 1.000 Prompts + 4 Modelle (597), OOF über die Prompt-Folds.

Aufruf (aus Name_That_LLM_2026/):  .venv/bin/python experiments/18_matched_readers/quick_test.py
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cv import get_folds  # noqa: E402

EXT = ROOT / "data" / "external" / "matched"
OUT = ROOT / "models" / "18_matched_readers"
FW = frozenset("a an the and or but if while although because as so than that which who whom whose this these those "
               "it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on "
               "at to from with without for of by over under between through into about is are was were be been being "
               "have has had do does did can could will would shall should may might must not no nor all any some each "
               "every both either neither other another such only also just very too more most less least here there "
               "when where why how what whether then now still yet even much many few little".split())
FOCUS = ["kimi-k3", "qwen3.8-max", "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash"]


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t or ""))).strip()


def shape_view(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_view(t):
    return " ".join(w.lower() if w.lower() in FW or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))


def tfidf_lr(fit_t, fit_y, fit_w, te_t):
    A, B = [], []
    for kw in (dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000),
               dict(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000)):
        v = TfidfVectorizer(sublinear_tf=True, lowercase=False, **kw).fit(fit_t)
        A.append(v.transform(fit_t)); B.append(v.transform(te_t))
    m = LogisticRegression(C=20, max_iter=3000).fit(sp.hstack(A).tocsr(), fit_y, sample_weight=fit_w)
    return m.predict_log_proba(sp.hstack(B).tocsr()), m.classes_


def svc4(fit_t, fit_y, fit_w, te_t):
    A, B = [], []
    for view, lo, hi, mf, wv in (("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7),
                                 ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)):
        prep = shape_view if view == "shape" else function_view if view == "function" else (lambda s: s)
        an = "char" if view == "shape" else "word" if view == "function" else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf,
                 dtype=np.float32)
        if an == "word":
            o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**o)
        A.append(v.fit_transform([prep(t) for t in fit_t]) * wv); B.append(v.transform([prep(t) for t in te_t]) * wv)
    m = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
    m.fit(sp.hstack(A, format="csr"), fit_y, sample_weight=fit_w)
    return m.decision_function(sp.hstack(B, format="csr")), m.classes_


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    tr["text"] = tr.text.fillna("").astype(str)
    labels = sorted(tr.model.unique())
    folds = get_folds(tr)
    mc = pd.read_parquet(EXT / "nlm_matched_clean" / "nlm_matched_clean.parquet")[["text", "model"]]
    mb = pd.concat([pd.read_csv(EXT / "markbarney5" / "passages.csv")[["text", "model"]],
                    pd.read_csv(EXT / "markbarney4" / "passages.csv")[["text", "model"]]], ignore_index=True)
    for d in (mc, mb):
        d["text"] = d.text.fillna("").astype(str)
    tr_keys = set(tr.text.map(norm))
    mc = mc[~mc.text.map(norm).isin(tr_keys) & mc.model.isin(labels)].drop_duplicates("text")
    mb = mb[~mb.text.map(norm).isin(tr_keys | set(mc.text.map(norm))) & mb.model.isin(labels)].drop_duplicates("text")
    print(f"train {len(tr)}, matched-clean {len(mc)}, markbarney {len(mb)}:", mb.model.value_counts().to_dict(), flush=True)

    y = tr.model.to_numpy()
    rows = []
    for reader in (tfidf_lr, svc4):
        for name, mb_w in (("ohne markbarney", 0.0), ("+ markbarney w0.5", 0.5), ("+ markbarney w1.0", 1.0)):
            t0 = time.time()
            pred = np.empty(len(tr), dtype=object)
            for k in np.unique(folds):
                tri, vai = folds != k, folds == k
                parts = [(tr.text[tri].tolist(), y[tri].tolist(), [1.0] * tri.sum()),
                         (mc.text.tolist(), mc.model.tolist(), [0.5] * len(mc))]
                if mb_w > 0:
                    parts.append((mb.text.tolist(), mb.model.tolist(), [mb_w] * len(mb)))
                ft = sum((p[0] for p in parts), []); fy = sum((p[1] for p in parts), []); fw = sum((p[2] for p in parts), [])
                s, cls = reader(ft, np.asarray(fy), np.asarray(fw, dtype=np.float32), tr.text[vai].tolist())
                pred[vai] = np.asarray(cls)[s.argmax(1)]
            rec = {c: float((pred[y == c] == c).mean()) for c in FOCUS}
            rows.append(dict(leser=reader.__name__, daten=name, bal=balanced_accuracy_score(y, pred), **rec,
                             sek=round(time.time() - t0)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "quick_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
