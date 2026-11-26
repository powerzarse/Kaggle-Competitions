"""Experiment 17, Gegenprobe auf echten Prefix-B-Daten: PB-115-Kopf auf train vs. train + Prefix-B-Pool.

train-Merkmale genau wie im 0.724-Notebook aus `lumibee/ntl-prefixb-full-115` (flash/glm53_features.csv,
paired_token_features.parquet), Pool-Merkmale aus unserem Datensatz (`prep_pool_dataset.py`, Dateien `pool_*`),
Matrix wie `_ours_add_pool` in V22. Kopf, Gewichte und Kette aus `simulate_pool.py` (OOF über die Prompt-Folds,
Ersatz-Stack → Auswahl nach GLM-Masse → Hybrid α 0.35 → optional Apex-ähnlich).

Aufruf (aus Name_That_LLM_2026/):
    DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs \
        .venv/bin/python experiments/17_pb115_pool/simulate_pb.py <lumibee-pb115-dir> [<pool-dataset-dir>]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simulate_pool as sp  # noqa: E402  (setzt sys.path für src/, 09, v1src)

pvs, s9 = sp.pvs, sp.s9
SPEC = sp.SPEC
OUT = sp.OUT


def matrix(d, ids, prefix=""):
    feats = {t: pd.read_csv(d / f"{prefix}{t}_features.csv", dtype={"id": str}).set_index("id").loc[ids]
             for t in ("flash", "glm53")}
    X = np.hstack([pvs.build_54_matrix(feats, ids, SPEC["features_54"], SPEC["diff_features"]),
                   pd.read_parquet(d / f"{prefix}paired_token_features.parquet").loc[ids][SPEC["features_61"]]
                   .to_numpy(np.float64)])
    assert np.isfinite(X).all() and X.shape == (len(ids), 115)
    return X


def main(pb_dir, pool_dir):
    real = sp.load_train()
    real["id"] = real.id.astype(str)
    labels = sorted(real.model.unique())
    L = np.asarray(labels)
    y = real.model.to_numpy()
    folds = sp.get_folds(real)
    X = matrix(pb_dir, real.id.tolist())
    lab = pd.read_csv(pool_dir / "pool_labels.csv", dtype={"id": str})
    pids = lab.id.tolist()
    Xp = matrix(pool_dir, pids, "pool_")
    yp = lab.model.to_numpy().astype(str)
    assert set(yp) <= set(labels)
    print(f"train {len(y)}, Pool {len(pids)} Texte (Prefix B), Klassen:", lab.model.value_counts().to_dict(), flush=True)

    cache = OUT / "oof_heads_pb.npz"
    heads = {}
    if cache.exists():
        with np.load(cache) as c:
            heads = {k: c[k] for k in c.files}
    for name, w, bal in (("nur train (wie 0.724)", 0.0, False), ("train + Pool w0.5", 0.5, False),
                         ("train + Pool ausgeglichen w0.25", 0.25, True),
                         ("train + Pool ausgeglichen w0.5", 0.5, True),
                         ("train + Pool ausgeglichen w1.0", 1.0, True)):
        if name not in heads:
            print(name, flush=True)
            heads[name] = sp.oof(X, y, folds, labels, Xp, yp, w, bal)
            np.savez(cache, **heads)

    zl = log_softmax(s9.stack_proxy(real.id, labels), 1)
    grp = lambda p: np.where(np.isin(p, s9.GLM), p, "rest")
    idx = np.array([labels.index(c) for c in y])
    glm = np.isin(y, s9.GLM)
    rows = []
    for cover, apex_like in ((0.40, False), (0.30, True), (0.40, True)):
        ref = L[sp.chain(zl, heads["nur train (wie 0.724)"], labels, cover, apex_like)]
        for name, ll in heads.items():
            f = L[sp.chain(zl, ll, labels, cover, apex_like)]
            d, lo, hi, pg = s9.bootstrap(y, ref, f, real.prompt_id, n=1000)
            rec = lambda c: float((f[y == c] == c).mean())
            rows.append(dict(abdeckung=cover, apex=apex_like, kopf=name,
                             kopf_bal=balanced_accuracy_score(y, L[ll.argmax(1)]),
                             kopf_glm3=balanced_accuracy_score(grp(y), grp(L[ll.argmax(1)])),
                             nll_glm=-ll[glm, idx[glm]].mean(),
                             kette=balanced_accuracy_score(y, f), d_pp=100 * d, lo=100 * lo, hi=100 * hi, p_gt0=pg,
                             rec_glm53=rec("glm-5.3"), rec_flash=rec("glm-5.3-flash"), rec_fable=rec("claude-fable-5"),
                             geaendert=int((f != ref).sum())))
        print(f"Abdeckung {cover:.0%}: Stack allein {balanced_accuracy_score(y, L[zl.argmax(1)]):.4f}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nΔ in Pp. gegen 'nur train' bei gleicher Abdeckung (Bootstrap 90 % über Prompts):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(OUT / "results_pb.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "dataset")
