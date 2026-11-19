"""Experiment 12: LLM-Familien-Spezialist (Kaggle-Ausgabe train_spec_logits.parquet) auswerten.

Vergleicht innerhalb der Familien (train.csv, nicht im Training) mit Ersatz-Stack und TF-IDF-Spezialist und
simuliert die V6-Kette (GLM 20 %, Apex) mit der Evidenz LR x LLM^t, LR x TF-IDF^t und LR x (LLM x TF-IDF)^t.

Aufruf: .venv/bin/python experiments/12_family_specialist/eval_llm_spec.py models/12_family_specialist/llm4b/train_spec_logits.parquet
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score as bas

import simulate as s9
from src.data import load_train


def main(path):
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    fams = res_mod.family_columns(labels)
    z = pd.read_parquet(path).astype({"id": str}).set_index("id").loc[real.id.astype(str)]
    llm = np.ones((len(real), len(labels)))
    for cols in fams[:3]:
        names = [labels[i] for i in cols]
        llm[:, cols] = softmax(z[names].to_numpy(np.float64), 1)
    tfidf = np.load(ROOT / "models/12_family_specialist/spec_tfidf_train_final.npy")
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    print("Innerhalb der Familie (train.csv):")
    for fam, cols in zip(("claude", "gpt", "gemini"), fams[:3]):
        m = np.isin(y, L[cols])
        f = lambda e: (L[cols][e[m][:, cols].argmax(1)] == y[m]).mean()
        print(f"  {fam:7s} Stack {f(zl):.3f} | TF-IDF {f(tfidf):.3f} | LLM {f(llm):.3f} | LLM x TF-IDF {f(llm * tfidf):.3f} "
              f"| Stack+LLM {f(zl + np.log(llm)):.3f}")
    post, _ = s9.glm_stage(zl, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
    stage = L[post.argmax(1)]
    ref, _ = apex.apply_arrays(zl, ev, labels, fams, texts, stage, 0.10, False, 0.0)
    ref = ref.astype(str)
    rows, finals = [], {}
    for name, e in (("TF-IDF", tfidf), ("LLM", llm), ("LLM x TF-IDF", llm * tfidf)):
        for t in (0.5, 1.0, 1.5):
            for s in (0.15, 0.25, 0.35, 0.5, 0.7):
                f, _ = apex.apply_arrays(zl, ev * e ** t, labels, fams, texts, stage, s, False, 0.0)
                f = f.astype(str)
                finals[(name, t, s)] = f
                rows.append(dict(evidenz=name, t=t, s=s, d=100 * (bas(y, f) - bas(y, ref)), changed=int((f != ref).sum())))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for name in df.evidenz.unique():
        print(f"\nΔ bal. acc in Pp. gegen V6, Evidenz LR x {name}^t (Zeilen t, Spalten Stärke):")
        print(df[df.evidenz == name].pivot(index="t", columns="s", values="d").round(2).to_string())
    best = df.sort_values("d", ascending=False).head(5)
    print("\nbeste:"); print(best.to_string(index=False))
    for _, r in best.head(3).iterrows():
        m, lo, hi, pp = s9.bootstrap(y, ref, finals[(r.evidenz, r.t, r.s)], real.prompt_id, n=1000)
        print(f"  {r.evidenz} t {r.t} s {r.s}: {100 * m:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}] P>0 {pp:.2f}")
    df.to_csv(Path(path).parent / "eval_chain.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
