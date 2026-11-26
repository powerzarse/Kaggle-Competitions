"""Experiment 18, Vortest 2: svc4-lite mit gespeicherten OOF-Scores und priorunabhängigen Maßen.

Vortest 1 (`quick_test.py`): markbarney hebt Recall Kimi/Qwen/Gemini-3.7 stark, bal. Acc. bleibt aber gleich
(Verschiebung zwischen Klassen?). Hier: Macro-AUC (one-vs-rest), bal. Acc. nach Sinkhorn-Ausgleich (gleiche
Klassenmasse auf den OOF-Zeilen), Verwechslungs-Deltas; Varianten ohne / alle markbarney w0.5 / nur Kimi+Qwen w0.5.

Aufruf (aus Name_That_LLM_2026/):  .venv/bin/python experiments/18_matched_readers/quick_test2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quick_test as q  # noqa: E402

FOCUS = q.FOCUS + ["gemini-3.6-flash", "gemini-3.1-pro-preview", "gpt-5.5"]


def sinkhorn(p, iters=50):
    p = p.copy()
    for _ in range(iters):
        p /= p.sum(0, keepdims=True)
        p /= p.sum(1, keepdims=True)
    return p


def main():
    tr = pd.read_csv(q.ROOT / "data" / "train.csv")
    tr["text"] = tr.text.fillna("").astype(str)
    labels = sorted(tr.model.unique())
    folds = q.get_folds(tr)
    mc = pd.read_parquet(q.EXT / "nlm_matched_clean" / "nlm_matched_clean.parquet")[["text", "model"]]
    mb = pd.concat([pd.read_csv(q.EXT / "markbarney5" / "passages.csv")[["text", "model"]],
                    pd.read_csv(q.EXT / "markbarney4" / "passages.csv")[["text", "model"]]], ignore_index=True)
    for d in (mc, mb):
        d["text"] = d.text.fillna("").astype(str)
    keys = set(tr.text.map(q.norm))
    mc = mc[~mc.text.map(q.norm).isin(keys) & mc.model.isin(labels)].drop_duplicates("text")
    mb = mb[~mb.text.map(q.norm).isin(keys | set(mc.text.map(q.norm))) & mb.model.isin(labels)].drop_duplicates("text")
    y = tr.model.to_numpy()
    yi = np.array([labels.index(c) for c in y])
    q.OUT.mkdir(parents=True, exist_ok=True)
    cache = q.OUT / "oof_svc4.npz"
    S = dict(np.load(cache)) if cache.exists() else {}
    variants = {"ohne markbarney": None, "+ markbarney w0.5": mb,
                "+ nur Kimi+Qwen w0.5": mb[mb.model.isin(["kimi-k3", "qwen3.8-max"])]}
    for name, extra in variants.items():
        if name in S:
            continue
        sc = np.zeros((len(tr), len(labels)), dtype=np.float32)
        for k in np.unique(folds):
            tri, vai = folds != k, folds == k
            ft = tr.text[tri].tolist() + mc.text.tolist()
            fy = y[tri].tolist() + mc.model.tolist()
            fw = [1.0] * tri.sum() + [0.5] * len(mc)
            if extra is not None:
                ft += extra.text.tolist(); fy += extra.model.tolist(); fw += [0.5] * len(extra)
            s, cls = q.svc4(ft, np.asarray(fy), np.asarray(fw, dtype=np.float32), tr.text[vai].tolist())
            sc[vai] = s[:, [list(cls).index(c) for c in labels]]
        S[name] = sc
        np.savez(cache, **S)
        print("fertig:", name, flush=True)

    L = np.asarray(labels)
    rows, preds = [], {}
    for name, sc in S.items():
        p = softmax(sc.astype(np.float64) * 12.0, axis=1)        # Blendgewicht 12 wie im Kern (log_softmax * 12)
        pb = sinkhorn(p)
        preds[name] = L[pb.argmax(1)]
        rows.append(dict(daten=name, bal=balanced_accuracy_score(y, L[sc.argmax(1)]),
                         bal_sinkhorn=balanced_accuracy_score(y, preds[name]),
                         macro_auc=roc_auc_score(yi, p, multi_class="ovr", average="macro"),
                         nll=-np.log(np.clip(p[np.arange(len(y)), yi], 1e-12, 1)).mean(),
                         **{c: float((preds[name][y == c] == c).mean()) for c in FOCUS}))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    print("\nRecall-Spalten nach Sinkhorn-Ausgleich:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(q.OUT / "quick_test2.csv", index=False)
    ref = preds["ohne markbarney"]
    for name in list(S)[1:]:
        f = preds[name]
        ch = pd.crosstab(pd.Series(y[ref != f], name="wahr"), pd.Series(np.char.add(np.char.add(ref[ref != f].astype(str), " -> "), f[ref != f].astype(str)), name="vorher -> nachher"))
        good = (f == y) & (ref != y); bad = (f != y) & (ref == y)
        print(f"\n{name}: {good.sum()} neu richtig, {bad.sum()} neu falsch")
        print("  neu richtig je Klasse:", pd.Series(y[good]).value_counts().head(8).to_dict())
        print("  neu falsch je Klasse:", pd.Series(y[bad]).value_counts().head(8).to_dict())
        print("  neu falsch: wohin:", pd.Series(f[bad]).value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
