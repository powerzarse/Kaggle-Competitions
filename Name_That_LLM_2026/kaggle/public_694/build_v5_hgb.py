"""Erzeugt kaggle/public_694/v5_hgb/ – V4 (ohne GSQ) + neuer GLM-Likelihood-Klassifikator.

Statt der LogReg aus dem öffentlichen Notebook (nur train.csv) nutzt die GLM-Stufe ein Gradient Boosting
(HistGradientBoosting) auf train.csv + 5.670 Open-Pool-Texten (Experiment 08). Das Modell wird im Abgabe-Notebook
aus den Merkmalen trainiert (kein Pickle -> keine sklearn-Versionsprobleme); Auswahl, Temperatur und
Abbruchgrenze der Stufe bleiben unverändert.

Die Trainingsmerkmale (~1,8 MB) liegen im privaten Datensatz linuszarse/ntl-glm-ll-train, weil Kaggle den
Notebook-Quelltext auf 1 MB begrenzt.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_694/build_v5_hgb.py --W 1.0 --pool-weight 0.5
    .venv/bin/kaggle datasets create -p kaggle/public_694/v5_hgb_data      # bzw. datasets version -m ...
    .venv/bin/kaggle kernels push -p kaggle/public_694/v5_hgb
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "public_694" / "v4_nogsq"
OUT = ROOT / "kaggle" / "public_694" / "v5_hgb"
DATA_DIR = ROOT / "kaggle" / "public_694" / "v5_hgb_data"
DATASET = "linuszarse/ntl-glm-ll-train"
MODEL = json.loads((ROOT / "experiments" / "08_glm_likelihood" / "v1src" / "ll_model.json").read_text())
HGB_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                  min_samples_leaf=40, early_stopping=False)

OLD_Z = ("    z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T "
         "+ np.asarray(model['intercept'])\n    return log_softmax(z, 1)")
NEW_Z = ("    if model.get('kind') == 'hgb':   # ours: boosted trees on train + Open-Pool likelihood features, fitted here\n"
         "        from sklearn.ensemble import HistGradientBoostingClassifier\n"
         "        with np.load(model['train_file'], allow_pickle=False) as d:\n"
         "            clf = HistGradientBoostingClassifier(random_state=0, **model['hgb_params']).fit(\n"
         "                d['X'], d['y'].astype(str), sample_weight=d['w'])\n"
         "        p = clf.predict_proba(X)[:, [list(clf.classes_).index(c) for c in model['classes']]]\n"
         "        return np.log(np.clip(p, 1e-9, 1))\n" + OLD_Z)


def training_data(pool_weight):
    tr = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    F = pd.read_parquet(ROOT / "models" / "08_glm_likelihood" / "train_v3" / "train_feats_L160.parquet")
    P = pd.read_parquet(ROOT / "models" / "08_glm_likelihood" / "pool_feats_L160.parquet")
    X_tr = F.loc[tr["id"], MODEL["features"]].to_numpy(np.float32)
    cnt = P["model"].value_counts()
    w_pool = pool_weight * (cnt.mean() / cnt.reindex(P["model"]).to_numpy())   # Klassen im Pool ausgleichen
    X = np.vstack([X_tr, P[MODEL["features"]].to_numpy(np.float32)])
    y = np.concatenate([tr["model"].to_numpy(), P["model"].to_numpy()]).astype("U32")
    w = np.concatenate([np.ones(len(tr)), w_pool]).astype(np.float32)
    assert np.isfinite(X).all() and len(X) == len(y) == len(w)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATA_DIR / "ll_train.npz", X=X, y=y, w=w)
    (DATA_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "NTL GLM LL Train", "id": DATASET, "licenses": [{"name": "CC-BY-SA-4.0"}]}, indent=2))
    return len(X)


def build(W, pool_weight):
    src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    nb = json.loads((SRC / "v4_nogsq.ipynb").read_text())
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    stage = [c for c in nb["cells"] if src(c).startswith("%%writefile /kaggle/temp/v1/v1_stage.py")]
    assert len(stage) == 1 and src(stage[0]).count(OLD_Z) == 1, "v1_stage.py hat sich geändert"
    stage[0]["source"] = src(stage[0]).replace(OLD_Z, NEW_Z)

    n = training_data(pool_weight)
    model_cell = [i for i, c in enumerate(nb["cells"]) if "ll_model.json').write_text" in src(c)]
    assert len(model_cell) == 1
    patch = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source":
             "# ours: GLM-Likelihood-Klassifikator = Gradient Boosting auf train.csv + Open Pool (Experiment 08)\n"
             "import json\nfrom pathlib import Path\n"
             "_hits = sorted({p.resolve() for g in ('ntl-glm-ll-train/ll_train.npz', '*/linuszarse/ntl-glm-ll-train/ll_train.npz')\n"
             "                for p in Path('/kaggle/input').glob(g)})\n"
             "assert len(_hits) == 1, f'll_train.npz not found exactly once: {_hits}'\n"
             "_m = json.loads(Path('/kaggle/temp/v1/ll_model.json').read_text())\n"
             f"_m.update(kind='hgb', train_file=str(_hits[0]), hgb_params={HGB_PARAMS!r}, W={W},\n"
             f"          source='ours: HGB train+pool, pool_weight {pool_weight}, {n} rows')\n"
             "Path('/kaggle/temp/v1/ll_model.json').write_text(json.dumps(_m))\n"
             "print('LL-Modell:', _m['kind'], _m['W'], _m['source'])"}
    nb["cells"].insert(model_cell[0] + 1, patch)
    nb["cells"][0]["source"] = ("# Name That LLM – Fork 0.694, Variante V5: ohne GSQ + neuer GLM-Likelihood-Klassifikator\n\n"
        "Wie V4 (GSQ-27B-Stufe aus, wegen 9-h-Limit). Zusätzlich nutzt die GLM-Stufe statt der LogReg (nur train.csv) "
        f"ein Gradient Boosting auf train.csv + 5.670 Open-Pool-Texten, bewertet unter GLM-5.3 und GLM-5.3-Flash "
        f"(eigener Kaggle-Lauf). Mischgewicht W = {W}. Auswahl, Temperatur und Abbruchgrenze der Stufe unverändert.\n\n"
        "Basis: lightningv08 *0.694 | GLM-5.3 Likelihood + Qwen3.8-27B LoRA* (Apache 2.0); GLM-Stufe und Merkmale von "
        "José Padilla (estopadilla); Pfad-Shim von flexonafft; alle Credits unten erhalten.")
    OUT.mkdir(exist_ok=True)
    (OUT / "v5_hgb.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.update(id="linuszarse/ntl-694-fork-v5-hgb", title="NTL 694 Fork v5 hgb", code_file="v5_hgb.ipynb",
                dataset_sources=meta["dataset_sources"] + [DATASET])
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"geschrieben: {OUT}/v5_hgb.ipynb ({(OUT / 'v5_hgb.ipynb').stat().st_size / 2**20:.1f} MB), {n} Trainingszeilen, W={W}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--pool-weight", type=float, default=1.0)
    a = ap.parse_args()
    build(a.W, a.pool_weight)
