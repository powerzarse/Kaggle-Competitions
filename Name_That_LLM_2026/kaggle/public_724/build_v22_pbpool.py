"""Erzeugt kaggle/public_724/v22_pbpool/ – V21 (lumibee 0.724 unverändert) + PB-115-Kopf auf train + Open Pool.

Experiment 17: Der PB-115-Kopf der GLM-Stufe wird im Original nur auf den 2.699 train-Texten trainiert. V22 hängt
unsere Prefix-B-bewerteten Pool-Texte an (privater Datensatz linuszarse/ntl-pb115-pool, Kernel
ntl-glm-scores-poolpb160), mit klassenweise ausgeglichenen Gewichten (jede Klasse dieselbe Gesamtmasse, Mittel
POOL_W). Simulation auf den alten Prefix-A-Läufen: +0,45 Pp. (40 % Abdeckung), ohne Ausgleich −0,3.

Änderungen, sonst nichts:
- Zelle mit `%%writefile /kaggle/working/prefix_variant_stage.py`: neue Funktion `_ours_add_pool` vor
  `fit_and_predict_pb115`; dort Pool-Zeilen anhängen und `sample_weight` an LR und LGB. Fehlt der Datensatz, läuft
  die Stufe wie V21 (nur train).
- FAST_COMMIT-Smoke der Stufe: zusätzlich prüfen, dass der Pool wirklich geladen wurde.
- kernel-metadata.json: Datensatz linuszarse/ntl-pb115-pool, neue ID/Titel.

Dateinamen im Datensatz mit Präfix `pool_`, weil das Notebook PB115_DIR über das erste
`paired_token_features.parquet` unter /kaggle/input findet.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_724/build_v22_pbpool.py [--pool-w 0.5]
    .venv/bin/kaggle kernels push -p kaggle/public_724/v22_pbpool
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v21_fork" / "v21_fork.ipynb"
DATASET = "linuszarse/ntl-pb115-pool"
STAGE_MARK = "%%writefile /kaggle/working/prefix_variant_stage.py"
SMOKE_MARK = "    _ll_pb, _m3_pb = _v1s.fit_and_predict_pb115("

ADD_POOL = '''OURS_POOL_W = {pool_w}
_OURS_POOL_USED = 0


def _ours_add_pool(X_tr, y, w_tr, spec, l2i):
    """ours (V22): Open-Pool texts scored with Prefix B (our Kaggle run, private dataset {dataset}),
    class-balanced weights (every class the same total mass, mean OURS_POOL_W). Without the dataset: train only."""
    global _OURS_POOL_USED
    hits = sorted(glob.glob('/kaggle/input/**/pool_paired_token_features.parquet', recursive=True))
    if not hits:
        print('V22 pool: dataset not found, PB-115 on train only', flush=True)
        return X_tr, y, w_tr
    d = Path(hits[0]).parent
    lab = pd.read_csv(d / 'pool_labels.csv', dtype={{'id': str}})
    ids = lab['id'].tolist()
    feats = {{t: pd.read_csv(d / f'pool_{{t}}_features.csv', dtype={{'id': str}}).set_index('id').loc[ids]
             for t in ('flash', 'glm53')}}
    Xp = np.hstack([build_54_matrix(feats, ids, spec['features_54'], spec['diff_features']),
                    pd.read_parquet(d / 'pool_paired_token_features.parquet').loc[ids][spec['features_61']]
                    .to_numpy(np.float64)])
    yp = lab['model'].map(l2i)
    assert yp.notna().all(), 'V22 pool: unknown labels'
    yp = yp.to_numpy(dtype=np.int64)
    assert np.isfinite(Xp).all() and Xp.shape == (len(ids), X_tr.shape[1])
    cls, cnt = np.unique(yp, return_counts=True)
    per = dict(zip(cls, len(yp) / (len(cls) * cnt)))
    wp = OURS_POOL_W * np.array([per[c] for c in yp])
    _OURS_POOL_USED = len(ids)
    print(f'V22 pool: {{len(ids)}} texts, {{len(cls)}} classes, weights {{wp.min():.2f}}-{{wp.max():.2f}}', flush=True)
    return np.vstack([X_tr, Xp]), np.concatenate([y, yp]), np.concatenate([w_tr, wp])


'''


def patch_stage(src, pool_w):
    head = "def fit_and_predict_pb115("
    assert src.count(head) == 1
    src = src.replace(head, ADD_POOL.format(pool_w=pool_w, dataset=DATASET) + head)
    old = "    assert np.isfinite(X_tr).all() and X_tr.shape == (len(tr_ids), 115)\n"
    assert src.count(old) == 1
    src = src.replace(old, old + "    X_tr, y, w_tr = _ours_add_pool(X_tr, y, np.ones(len(y)), spec, l2i)   # ours (V22)\n")
    n = src.count(".fit(X_tr_s, y)")
    assert n == 2, n                       # LR18 und LGB18 (y3 wird im Original nicht genutzt)
    src = src.replace(".fit(X_tr_s, y)", ".fit(X_tr_s, y, sample_weight=w_tr)")
    return src


def build(pool_w):
    nb = json.loads(SRC.read_text())
    cells = nb["cells"]
    stage = [i for i, c in enumerate(cells) if "".join(c["source"]).startswith(STAGE_MARK)]
    smoke = [i for i, c in enumerate(cells) if SMOKE_MARK in "".join(c["source"])]
    assert len(stage) == 1 and len(smoke) == 1, (stage, smoke)
    s = "".join(cells[stage[0]]["source"])
    cells[stage[0]]["source"] = patch_stage(s, pool_w)
    s = "".join(cells[smoke[0]]["source"])
    line_end = s.index("\n", s.index(SMOKE_MARK))
    s = s[:line_end + 1] + ("    assert _v1s._OURS_POOL_USED > 0, 'V22: pool rows not loaded'   # ours (V22)\n"
                            "    print('V22 pool rows in PB-115 head:', _v1s._OURS_POOL_USED, flush=True)\n") + s[line_end + 1:]
    cells[smoke[0]]["source"] = s
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     "**V22 (private fork):** lumibee's 0.724 notebook, unchanged except that the PB-115 GLM head is "
                     f"trained on train.csv plus Open-Pool texts scored with the same Prefix B (dataset `{DATASET}`, "
                     f"class-balanced weights, mean {pool_w}). All credit for the pipeline goes to the authors below."})
    out = HERE / "v22_pbpool"
    out.mkdir(exist_ok=True)
    (out / "v22_pbpool.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = json.loads((HERE / "v21_fork" / "kernel-metadata.json").read_text())
    meta.update(id="linuszarse/ntl-724-fork-v22-pbpool", title="NTL 724 Fork v22 pbpool", code_file="v22_pbpool.ipynb")
    if DATASET not in meta["dataset_sources"]:
        meta["dataset_sources"].append(DATASET)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", out, "| Pool-Gewicht", pool_w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-w", type=float, default=0.5)
    build(ap.parse_args().pool_w)
