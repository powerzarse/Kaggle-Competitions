"""Experiment 18, Test 3: echtes svc4 (mit Pool) ± markbarney, OOF über die Prompt-Folds auf train.

Vektorisierer fest aus dem Kern des 0.727-Notebooks (`ntl666-prefit/core/svc4/vectorizer-*.joblib`), nur die
LinearSVC (C 0.5, balanced) wird neu gefittet – so wie später nur `model.npz` getauscht würde.
Daten je Fold: train-Folds w1 + nlm-matched-clean w0.5 + Open Pool v14 (lokal; Kern: v13) ohne train/matched-clean,
nach Klassen geschichtete Stichprobe POOL_FRAC mit Gewicht 0.35 / POOL_FRAC (gleiche Masse je Klasse wie der volle
Pool; voller Pool passt nicht in 16 GB) ± markbarney w0.5.
Auswertung wie Vortest 2 (Sinkhorn auf softmax(12 · Score)) + Bootstrap über Prompts.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/18_matched_readers/svc4_pool_test.py <ntl666-prefit-dir>
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quick_test as q  # noqa: E402
from quick_test2 import sinkhorn  # noqa: E402

POOL_FRAC = 0.30
FOCUS = ["kimi-k3", "qwen3.8-max", "gemini-3.7-flash", "gemini-3.8-flash", "gpt-5.5", "gpt-5.6-sol"]


def views(prefit, texts):
    meta = [("vectorizer-0.joblib", lambda s: s, 1.0), ("vectorizer-1.joblib", lambda s: s, 0.7),
            ("vectorizer-2.joblib", q.shape_view, 0.5), ("vectorizer-3.joblib", q.function_view, 0.5)]
    out = []
    for f, prep, w in meta:
        v = joblib.load(prefit / "core" / "svc4" / f)
        out.append(v.transform([prep(t) for t in texts]) * w)
    return sp.hstack(out, format="csr", dtype=np.float32)


def boot(y, a, b, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.factorize(groups)[0]
    idx = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    d = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
        d.append(balanced_accuracy_score(y[s], b[s]) - balanced_accuracy_score(y[s], a[s]))
    d = np.asarray(d)
    return np.percentile(d, 5), np.percentile(d, 95), (d > 0).mean()


def main(prefit):
    tr = pd.read_csv(q.ROOT / "data" / "train.csv")
    tr["text"] = tr.text.fillna("").astype(str)
    labels = sorted(tr.model.unique())
    folds = q.get_folds(tr)
    mc = pd.read_parquet(q.EXT / "nlm_matched_clean" / "nlm_matched_clean.parquet")[["text", "model"]]
    mb = pd.concat([pd.read_csv(q.EXT / "markbarney5" / "passages.csv")[["text", "model"]],
                    pd.read_csv(q.EXT / "markbarney4" / "passages.csv")[["text", "model"]]], ignore_index=True)
    pool = pd.read_parquet(q.ROOT / "data/external/open_pool/ntl_open_pool.parquet", columns=["text", "model"])
    for d in (mc, mb, pool):
        d["text"] = d.text.fillna("").astype(str)
    keys = set(tr.text.map(q.norm))
    mc = mc[~mc.text.map(q.norm).isin(keys) & mc.model.isin(labels)].drop_duplicates("text")
    keys |= set(mc.text.map(q.norm))
    pool = pool[pool.model.isin(labels)]
    pool = pool[~pool.text.map(q.norm).isin(keys)].drop_duplicates("text")
    keys |= set(pool.text.map(q.norm))
    mb = mb[~mb.text.map(q.norm).isin(keys) & mb.model.isin(labels)].drop_duplicates("text")
    pool = pool.groupby("model", group_keys=False).sample(frac=POOL_FRAC, random_state=0)
    print(f"train {len(tr)}, matched-clean {len(mc)}, Pool-Stichprobe {len(pool)} (w {0.35 / POOL_FRAC:.3f}), "
          f"markbarney {len(mb)}", flush=True)

    t0 = time.time()
    X_tr, X_mc, X_pool, X_mb = (views(prefit, d.text.tolist()) for d in (tr, mc, pool, mb))
    print(f"Merkmale fertig in {time.time() - t0:.0f} s, nnz {sum(M.nnz for M in (X_tr, X_mc, X_pool, X_mb)) / 1e6:.0f} M",
          flush=True)
    y = tr.model.to_numpy()
    yi = np.array([labels.index(c) for c in y])
    kq = mb.model.isin(["kimi-k3", "qwen3.8-max"]).to_numpy()
    variants = {"ohne markbarney": None, "+ nur Kimi+Qwen w0.5": kq, "+ markbarney w0.5": np.ones(len(mb), bool)}
    cache = q.OUT / "oof_svc4_pool.npz"
    S = dict(np.load(cache)) if cache.exists() else {}
    for name, sel in variants.items():
        if name in S:
            continue
        sc = np.zeros((len(tr), len(labels)), dtype=np.float32)
        for k in np.unique(folds):
            t1 = time.time()
            tri, vai = folds != k, folds == k
            Xs = [X_tr[tri], X_mc, X_pool]
            ys = [y[tri], mc.model.to_numpy(), pool.model.to_numpy()]
            ws = [np.ones(tri.sum()), np.full(len(mc), 0.5), np.full(len(pool), 0.35 / POOL_FRAC)]
            if sel is not None:
                Xs.append(X_mb[sel]); ys.append(mb.model.to_numpy()[sel]); ws.append(np.full(sel.sum(), 0.5))
            m = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
            m.fit(sp.vstack(Xs, format="csr"), np.concatenate(ys), sample_weight=np.concatenate(ws))
            s = m.decision_function(X_tr[vai])
            sc[vai] = s[:, [list(m.classes_).index(c) for c in labels]]
            print(f"  {name} Fold {k}: {time.time() - t1:.0f} s", flush=True)
        S[name] = sc
        np.savez(cache, **S)

    L = np.asarray(labels)
    preds, rows = {}, []
    for name, sc in S.items():
        p = softmax(sc.astype(np.float64) * 12.0, axis=1)
        preds[name] = L[sinkhorn(p).argmax(1)]
        rows.append(dict(daten=name, bal=balanced_accuracy_score(y, L[sc.argmax(1)]),
                         bal_sinkhorn=balanced_accuracy_score(y, preds[name]),
                         macro_auc=roc_auc_score(yi, p, multi_class="ovr", average="macro"),
                         **{c: float((preds[name][y == c] == c).mean()) for c in FOCUS}))
    ref = preds["ohne markbarney"]
    for r in rows:
        lo, hi, pg = boot(y, ref, preds[r["daten"]], tr.prompt_id)
        r.update(d_pp=100 * (r["bal_sinkhorn"] - rows[0]["bal_sinkhorn"]), lo=100 * lo, hi=100 * hi, p_gt0=pg)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print("\nRecall-Spalten nach Sinkhorn; Δ in Pp. auf bal_sinkhorn, Bootstrap 90 % über Prompts:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(q.OUT / "svc4_pool_test.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
