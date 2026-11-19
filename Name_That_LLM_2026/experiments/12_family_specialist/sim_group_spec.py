"""Experiment 12b: Gruppen-Spezialisten für Verwechslungen über Familiengrenzen (Apex greift dort nicht).

Gruppen: kimi = {kimi-k3, gpt-5.5, qwen3.8-max}, fable = {claude-fable-5, glm-5.3, glm-5.3-flash}.
TF-IDF + LR (wie export_spec.py) auf Open-Pool-Texten der Gruppe (bis 15k je Klasse), bewertet auf train.csv.
Nach der V6-Kette (GLM 20 %, Apex): Texte, deren Vorhersage in der Gruppe liegt, innerhalb der Gruppe neu
entscheiden mit log p_Kette + t * log Spezialist (p_Kette: Logits nach der GLM-Stufe, gekappt auf die Gruppe).

Aufruf: .venv/bin/python experiments/12_family_specialist/sim_group_spec.py
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
from sklearn.metrics import balanced_accuracy_score as bas

import simulate as s9
from src.data import load_train

GROUPS = {"kimi": ["gpt-5.5", "kimi-k3", "qwen3.8-max"], "fable": ["claude-fable-5", "glm-5.3", "glm-5.3-flash"]}
OUT = ROOT / "models" / "12_family_specialist"


def main():
    t0 = time.time()
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    fams = res_mod.family_columns(labels)
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["text", "model"])
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
    ref, _ = apex.apply_arrays(zl, ev, labels, fams, texts, L[post.argmax(1)], 0.10, False, 0.0)
    ref = ref.astype(str)
    print(f"V6-Kette: {bas(y, ref):.4f}", flush=True)
    specs = {}
    for g, cls in GROUPS.items():
        cache = OUT / f"group_{g}_train.npy"
        if cache.exists():
            specs[g] = np.load(cache)
            continue
        P = pd.concat([d.sample(min(len(d), 15000), random_state=0) for _, d in pool[pool.model.isin(cls)].groupby("model")])
        vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True)
        vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=400000, sublinear_tf=True)
        X = hstack([vw.fit_transform(P.text), vc.fit_transform(P.text)]).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced", solver="saga", random_state=0).fit(X, P.model)
        p = clf.predict_proba(hstack([vw.transform(texts), vc.transform(texts)]).tocsr())
        specs[g] = p[:, [list(clf.classes_).index(c) for c in cls]]
        np.save(cache, specs[g])
        print(f"{g}: {len(P)} Pool-Texte ({time.time() - t0:.0f} s)", flush=True)
    for g, cls in GROUPS.items():
        cols = [labels.index(c) for c in cls]
        m = np.isin(y, cls)
        sp = specs[g]
        print(f"\nGruppe {g} {cls}: innerhalb der Gruppe (wahre Klasse in der Gruppe, n={m.sum()})")
        print(f"  Ersatz-Stack {(L[cols][zl[m][:, cols].argmax(1)] == y[m]).mean():.3f} | nach GLM-Stufe "
              f"{(L[cols][post[m][:, cols].argmax(1)] == y[m]).mean():.3f} | Spezialist {(L[cols][sp[m].argmax(1)] == y[m]).mean():.3f}"
              f" | Stufe + Spezialist {(L[cols][(log_softmax(post[m][:, cols], 1) + np.log(sp[m])).argmax(1)] == y[m]).mean():.3f}")
        inG = np.isin(ref, cls)
        base = log_softmax(post[:, cols], 1)
        res = {}
        for t in (0.25, 0.5, 0.75, 1.0, 1.5):
            f = ref.copy()
            f[inG] = L[cols][(base + t * np.log(sp)).argmax(1)][inG]
            res[t] = f
            rec = pd.Series(f == y).groupby(y).mean()[cls].round(2).to_dict()
            print(f"  t {t}: {100 * (bas(y, f) - bas(y, ref)):+.2f} Pp. | anders {(f != ref).sum()} | Recall {rec}")
        tb = max(res, key=lambda t: bas(y, res[t]))
        mm, lo, hi, pp = s9.bootstrap(y, ref, res[tb], real.prompt_id, n=1000)
        print(f"  bestes t {tb}: {100 * mm:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}] P>0 {pp:.2f}")


if __name__ == "__main__":
    main()
