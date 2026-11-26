"""Baut den privaten Datensatz linuszarse/ntl-pb115-pool aus der Ausgabe von ntl-glm-scores-poolpb160.

Eingabe: Ordner `out/` des Kernels (z. B. per `kaggle kernels output linuszarse/ntl-glm-scores-poolpb160 -p <dir>`):
`c*_{flash,glm53}_features.csv`, `c*_{flash,glm53}_scores.npz`, `labels.csv` (mit Spalte source), `format_check.json`.
Ausgabe (`models/17_pb115_pool/dataset/`): `pool_flash_features.csv`, `pool_glm53_features.csv`,
`pool_paired_token_features.parquet` (61 + Einzelmerkmale, `v1_token_features` des 0.724-Notebooks),
`pool_labels.csv`, `dataset-metadata.json`. Die 40 train-Prüftexte werden nur zur Formatprüfung genutzt.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python experiments/17_pb115_pool/prep_pool_dataset.py <out-dir>
    .venv/bin/kaggle datasets create -p models/17_pb115_pool/dataset      # privat
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "v1src"))
import v1_token_features as vtf  # noqa: E402

OUT = ROOT / "models" / "17_pb115_pool" / "dataset"
PB115 = json.loads((HERE / "v1src" / "spec.json").read_text())


def main(run_dir):
    run = Path(run_dir)
    lab = pd.read_csv(run / "labels.csv", dtype={"id": str}).set_index("id")
    feats = {"flash": [], "glm53": []}
    paired = []
    chunks = sorted({int(p.name.split("_")[0][1:]) for p in run.glob("c*_glm53_scores.npz")})
    for ci in chunks:
        if not (run / f"c{ci}_flash_scores.npz").exists():
            continue
        for t in feats:
            feats[t].append(pd.read_csv(run / f"c{ci}_{t}_features.csv", dtype={"id": str}).set_index("id"))
        paired.append(vtf.extract_paired_score_features(run / f"c{ci}_flash_scores.npz", run / f"c{ci}_glm53_scores.npz"))
    F = {t: pd.concat(v) for t, v in feats.items()}
    P = pd.concat(paired)
    ids = [i for i in F["glm53"].index if i in F["flash"].index and i in P.index]
    check = [i for i in ids if lab.loc[i, "source"] == "train"]
    pool = [i for i in ids if lab.loc[i, "source"] == "pool"]
    assert not set(PB115["features_61"]) - set(P.columns), "fehlende gepaarte Merkmale"
    print(f"Häppchen {chunks}: {len(pool)} Pool-Texte, {len(check)} train-Prüftexte")
    print("Klassen:", lab.loc[pool, "model"].value_counts().to_dict())

    # Formatprüfung gegen lumibees train-Merkmale (gleicher Prefix → gleiche Tokenzahl)
    ref_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if ref_dir is not None:
        for t in ("flash", "glm53"):
            ref = pd.read_csv(ref_dir / f"{t}_features.csv", dtype={"id": str}).set_index("id").loc[check]
            ours = F[t].loc[check]
            print(t, "n_tok gleich", int((ours.n_tok == ref.n_tok).sum()), "/", len(check),
                  "| corr lp_mean", round(float(np.corrcoef(ours.lp_mean, ref.lp_mean)[0, 1]), 4))
        refp = pd.read_parquet(ref_dir / "paired_token_features.parquet").loc[check][PB115["features_61"]]
        c = [np.corrcoef(P.loc[check, k], refp[k])[0, 1] for k in PB115["features_61"] if refp[k].std() > 0]
        print("gepaarte Merkmale: Korrelation je Merkmal Median", round(float(np.median(c)), 3), "min", round(float(np.min(c)), 3))

    OUT.mkdir(parents=True, exist_ok=True)
    for t in F:
        F[t].loc[pool].reset_index().to_csv(OUT / f"pool_{t}_features.csv", index=False)
    P.loc[pool].to_parquet(OUT / "pool_paired_token_features.parquet")
    lab.loc[pool, ["model"]].reset_index().to_csv(OUT / "pool_labels.csv", index=False)
    assert np.isfinite(P.loc[pool][PB115["features_61"]].to_numpy(np.float64)).all()
    meta = {"title": "ntl-pb115-pool", "id": "linuszarse/ntl-pb115-pool", "licenses": [{"name": "CC-BY-SA-4.0"}],
            "description": "Open-Pool texts (crtproduction/name-that-llm-open-pool, CC BY-SA 4.0) scored under "
                           "GLM-5.3-Flash and GLM-5.3 with Prefix B (160 tokens), PB-115 feature format (lumibee)."}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
