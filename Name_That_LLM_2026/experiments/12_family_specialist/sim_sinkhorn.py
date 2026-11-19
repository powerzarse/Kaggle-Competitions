"""Sinkhorn-Stärke des Stacks (decide_scores aus infer.py) auf dem Ersatz-Stack, bei unausgewogenen Testmischungen.

decide_scores: (1-s) log P + s log Q, Q = Sinkhorn-Ausgleich von P auf target = (1-rho) * mean(P) + rho * prior(train).
V6 nutzt s = 0.30, rho = 0.35. Gemessen wird balanced accuracy auf Teilmengen von train.csv mit Klassenanteilen
aus Dirichlet(alpha) (alpha klein = stark unausgewogen), je 40 Ziehungen, gepaart über dieselben Ziehungen.
Danach die V6-Kette (GLM-Stufe 20 % + Apex) auf denselben Ziehungen für die besten Einstellungen.

Aufruf: .venv/bin/python experiments/12_family_specialist/sim_sinkhorn.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "09_postprocess"))

import numpy as np
import pandas as pd
from scipy.special import log_softmax, softmax
from sklearn.metrics import balanced_accuracy_score

import simulate as s9
from src.data import load_train


def decide_scores(P, s, rho, prior):
    Q = P.copy()
    marg = np.clip(P.mean(0), 1e-6, None)
    marg /= marg.sum()
    target = np.clip((1.0 - rho) * marg + rho * prior, 1e-6, None)
    target /= target.sum()
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - s) * np.log(P + 1e-12) + s * np.log(Q + 1e-12)


def main():
    res_mod, apex, v1, ll_model = s9.load_modules()
    real = load_train()
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    yi = np.array([labels.index(c) for c in y])
    texts = np.asarray(real.text.fillna("").astype(str).tolist(), dtype=object)
    P_all = softmax(s9.stack_proxy(real.id, labels), 1)
    prior = np.full(18, 1 / 18)
    with np.load(s9.OUT / "oof_cache.npz") as c:
        ll_glm, ev = c["ll"], c["ev"]
    families = res_mod.family_columns(labels)
    rng = np.random.default_rng(0)
    grid = [(s, rho) for s in (0.0, 0.3, 0.5, 0.7, 1.0) for rho in (0.0, 0.35, 0.7, 1.0)]
    for alpha in (100.0, 3.0, 1.0):
        draws = []
        for _ in range(40):
            w = rng.dirichlet(np.full(18, alpha))
            n = np.minimum(np.round(w / w.max() * 150).astype(int), 150)
            n = np.maximum(n, 5)
            idx = np.concatenate([rng.choice(np.flatnonzero(yi == k), min(n[k], (yi == k).sum()), replace=False)
                                  for k in range(18)])
            draws.append(idx)
        res = {}
        for s, rho in grid:
            res[(s, rho)] = np.array([balanced_accuracy_score(yi[i], decide_scores(P_all[i], s, rho, prior).argmax(1))
                                      for i in draws])
        base = res[(0.3, 0.35)]
        print(f"\nalpha {alpha} (Testgröße ~{np.mean([len(i) for i in draws]):.0f}): bal. acc des Stacks, Δ gegen V6 (0.3/0.35) in Pp.")
        tab = pd.DataFrame([dict(s=s, rho=rho, bal=v.mean(), d=100 * (v - base).mean(), d_lo=100 * np.quantile(v - base, .1))
                            for (s, rho), v in res.items()])
        print(tab.pivot(index="s", columns="rho", values="d").round(2).to_string())
        # Kette: GLM-Stufe 20 % + Apex für ausgewählte Einstellungen
        chain = {}
        for s, rho in ((0.3, 0.35), (0.5, 0.35), (0.7, 0.35), (0.5, 0.7), (0.0, 0.0)):
            vals = []
            for i in draws:
                zl = log_softmax(decide_scores(P_all[i], s, rho, prior), 1)
                post, _ = s9.glm_stage(zl, ll_glm[i], labels, ll_model["entropy_ref"], v1.match_temperature, 1.0, 0.20)
                fin, _ = apex.apply_arrays(zl, ev[i], labels, families, list(texts[i]), L[post.argmax(1)], 0.10, False, 0.0)
                vals.append(balanced_accuracy_score(y[i], fin.astype(str)))
            chain[(s, rho)] = np.array(vals)
        b = chain[(0.3, 0.35)]
        print("  V6-Kette:", ", ".join(f"s {s}/rho {r}: {100 * (v - b).mean():+.2f}" for (s, r), v in chain.items()))


if __name__ == "__main__":
    main()
