"""Experiment 13: Prior-Korrektur der Stack-Logits vor GLM-Stufe und Apex (V6-Kette, Ersatz-Stack auf train.csv).

In V6 setzt Apex jede Nicht-GLM-Vorhersage auf argmax(refine(Stack)) zurück; der Sinkhorn-Ausgleich des Stacks
wirkt dort nicht. Der Stack lernt die Klassenanteile des Open Pools (gemini-3.7-flash 3,5k gegen 3.8-flash 23k).
Getestet: (a) b_c = -alpha * log(Pool-Anteil_c), ein Parameter; (b) Bias je Klasse, Koordinatensuche auf einer
Prompt-Hälfte, bewertet auf der anderen (2-fach, mehrfach wiederholt).

Aufruf: .venv/bin/python experiments/13_prior_correction/sim_prior.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score as bas

import simulate as s9
from src.data import load_train


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    texts = real.text.fillna("").astype(str).tolist()
    fams = res_mod.family_columns(labels)
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    pool = pd.read_parquet(ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["model"]).model
    share = pool.value_counts(normalize=True).reindex(labels).to_numpy()
    logprior = np.log(share) - np.log(share).mean()

    def chain(b):
        z = log_softmax(zl + b, 1)
        post, _ = s9.glm_stage(z, ll_glm, labels, ll_model["entropy_ref"], v1.match_temperature, W=1.0, cover=0.20)
        f, _ = apex.apply_arrays(z, ev, labels, fams, texts, L[post.argmax(1)], 0.10, False, 0.0)
        return f.astype(str)

    ref = chain(np.zeros(len(labels)))
    print(f"V6-Kette: {bas(y, ref):.4f}; vorhergesagt je Klasse (wahr je 150):")
    print(pd.Series(ref).value_counts().reindex(labels).to_string())
    print("\n(a) b = -alpha * log Pool-Anteil")
    best = {}
    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        f = chain(-a * logprior)
        best[a] = f
        rec = pd.Series(f == y).groupby(y).mean()
        print(f"  alpha {a:.1f}: {100 * (bas(y, f) - bas(y, ref)):+.2f} Pp. | 3.7-flash {rec['gemini-3.7-flash']:.2f} "
              f"3.8-flash {rec['gemini-3.8-flash']:.2f} fable {rec['claude-fable-5']:.2f} | anders {(f != ref).sum()}")
    for a in (0.3, 0.5):
        m, lo, hi, pp = s9.bootstrap(y, ref, best[a], real.prompt_id, n=1000)
        print(f"  Bootstrap alpha {a}: {100 * m:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}] P>0 {pp:.2f}")

    print("\n(b) Bias je Klasse, Prompt-Kreuzvalidierung (fit auf Hälfte, Bewertung auf der anderen)")
    groups = pd.factorize(real.prompt_id)[0]
    grid = np.arange(-1.0, 1.01, 0.25)
    gains = []
    for rep in range(3):
        rng = np.random.default_rng(rep)
        half = rng.permutation(groups.max() + 1) < (groups.max() + 1) // 2
        A = half[groups]
        for fit_m, ev_m in ((A, ~A), (~A, A)):
            b = np.zeros(len(labels))
            score = lambda bb: bas(y[fit_m], chain(bb)[fit_m])
            cur = score(b)
            for sweep in range(2):
                for k in range(len(labels)):
                    for g in grid:
                        t = b.copy(); t[k] = g
                        sc = score(t)
                        if sc > cur + 1e-9:
                            cur, b = sc, t
            f = chain(b)
            d = 100 * (bas(y[ev_m], f[ev_m]) - bas(y[ev_m], ref[ev_m]))
            gains.append(d)
            nz = {labels[k]: b[k] for k in np.flatnonzero(b)}
            print(f"  Wdh {rep}: fit {100 * (cur - bas(y[fit_m], ref[fit_m])):+.2f}, Test-Hälfte {d:+.2f} | {nz}", flush=True)
    print(f"  Mittel Test-Hälfte: {np.mean(gains):+.2f} Pp. (min {np.min(gains):+.2f}, max {np.max(gains):+.2f})")


if __name__ == "__main__":
    main()
