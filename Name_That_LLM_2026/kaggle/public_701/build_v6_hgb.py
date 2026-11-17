"""Erzeugt kaggle/public_701/v6_hgb/ – öffentliches 0.701-Notebook + unser GLM-Likelihood-Klassifikator (V6).

Basis: alexchilton "Name That LLM 0.701: All Public + What Failed" (Linie lumibee Kiro v44), unverändert in
public_701.ipynb. Dessen GLM-Stufe (v1_stage.py, im Notebook als tar-Payload eingebettet) nutzt dieselbe LogReg
(ll_model.json identisch mit dem 0.694-Notebook). Wir ersetzen sie wie in V5 durch Gradient Boosting auf
train.csv + 5.670 Open-Pool-Texten (Experiment 08; Trainingsmerkmale im privaten Datensatz linuszarse/ntl-glm-ll-train).

Unterschied zu V5: Die Stufe hat hier eine Zeitgrenze und nur 180 s Reserve nach dem Scoring. Deshalb wird das
Boosting direkt nach dem Entpacken der Stufe trainiert und gepickelt (gleiche Python-Umgebung wie der
v1_stage-Subprozess); die Zeitplanung der Stufe rechnet die Trainingszeit so automatisch mit ein.
Im Commit-Lauf (FAST_COMMIT) wird zusätzlich member_logp der gepatchten Stufe an 5 Trainingszeilen geprüft.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v6_hgb.py --W 1.0
    .venv/bin/kaggle kernels push -p kaggle/public_701/v6_hgb
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public_694"))
from build_v5_hgb import DATASET, HGB_PARAMS, OLD_Z  # noqa: E402  (gleiche Ersetzungsstelle, gleiche Parameter)

OUT = HERE / "v6_hgb"
SETUP_MARK = "# ---- ours, V1 setup and checks"
NEW_Z = ("    if model.get('kind') == 'hgb':   # ours (V6): boosted trees on train + Open-Pool features, fitted in the notebook\n"
         "        import pickle\n"
         "        with open(model['clf_file'], 'rb') as f:\n"
         "            clf = pickle.load(f)\n"
         "        p = clf.predict_proba(X)[:, [list(clf.classes_).index(c) for c in model['classes']]]\n"
         "        return np.log(np.clip(p, 1e-9, 1))\n" + OLD_Z)


def patch_cell(W, pool_weight):
    return f'''# ours (V6): GLM likelihood classifier = gradient boosting on train.csv + 5,670 Open-Pool texts scored under
# GLM-5.3 and GLM-5.3-Flash (our own Kaggle run, features in the private dataset {DATASET}).
# Patches the extracted v1_stage.py (member_logp) and ll_model.json; selection, deadline, temperature and the
# max-changed guard of the stage are unchanged. Fitted here, before the stage computes its time budget.
import json, pickle, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
_t = time.time()
_OLD = {OLD_Z!r}
_NEW = {NEW_Z!r}
_stage = V1_DIR / 'v1_stage.py'
_src = _stage.read_text()
assert _src.count(_OLD) == 1, 'V6: member_logp in v1_stage.py is not the expected one'
_stage.write_text(_src.replace(_OLD, _NEW))
_hits = sorted({{p.resolve() for g in ('ntl-glm-ll-train/ll_train.npz', '*/linuszarse/ntl-glm-ll-train/ll_train.npz')
                for p in Path('/kaggle/input').glob(g)}})
assert len(_hits) == 1, f'V6: ll_train.npz not found exactly once: {{_hits}}'
_m = json.loads((V1_DIR / 'll_model.json').read_text())
with np.load(_hits[0], allow_pickle=False) as _d:
    _Xtr, _ytr, _wtr = _d['X'], _d['y'].astype(str), _d['w']
assert _Xtr.shape == (len(_ytr), len(_m['features'])) and sorted(set(_ytr)) == sorted(_m['classes'])
_clf = HistGradientBoostingClassifier(random_state=0, **{HGB_PARAMS!r}).fit(_Xtr, _ytr, sample_weight=_wtr)
with open(V1_DIR / 'll_hgb.pkl', 'wb') as _f:
    pickle.dump(_clf, _f)
_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W={W},
          source='ours: HGB train+pool, pool_weight {pool_weight}, ' + str(len(_ytr)) + ' rows')
(V1_DIR / 'll_model.json').write_text(json.dumps(_m))
print('V6 LL model:', _m['kind'], 'W', _m['W'], _m['source'], 'fit', round(time.time() - _t), 's', flush=True)
if FAST_COMMIT:   # commit only: the patched scoring code path on 5 train rows
    import importlib, sys
    sys.path.insert(0, str(V1_DIR))
    import v1_stage; importlib.reload(v1_stage)
    _X5 = _Xtr[:5].astype(np.float64)
    _feats = {{t: pd.DataFrame({{f.split(':', 1)[1]: _X5[:, j] for j, f in enumerate(_m['features']) if f.startswith(t + ':')}},
                              index=list(range(5))) for t in ('flash', 'glm53')}}
    _lp = v1_stage.member_logp(_m, _feats, list(range(5)))
    assert _lp.shape == (5, 18) and np.isfinite(_lp).all() and np.allclose(np.exp(_lp).sum(1), 1, atol=1e-6)
    print('V6 member_logp smoke OK, train labels', list(_ytr[:5]), '-> argmax',
          [_m['classes'][i] for i in _lp.argmax(1)], flush=True)
'''


def build(W, pool_weight):
    src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    nb = json.loads((HERE / "public_701.ipynb").read_text())
    meta = json.loads((HERE / "original-kernel-metadata.json").read_text())
    setup = [i for i, c in enumerate(nb["cells"]) if src(c).startswith(SETUP_MARK)]
    assert len(setup) == 1, "Setup-Zelle der GLM-Stufe nicht eindeutig gefunden"
    nb["cells"].insert(setup[0] + 1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                      "source": patch_cell(W, pool_weight)})
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source":
        "# Fork V6: 0.701 + GLM-Likelihood-Klassifikator (Gradient Boosting mit Open-Pool-Daten)\n\n"
        "Unverändert: alexchilton *Name That LLM 0.701: All Public + What Failed* (Linie lumibee *NTL Kiro v44 Apex "
        "Grandmaster*). Unsere einzige Änderung: Die GLM-Stufe (José Padilla) nutzt statt der LogReg auf train.csv ein "
        f"Gradient Boosting auf train.csv + 5.670 Open-Pool-Texten, bewertet unter GLM-5.3 und GLM-5.3-Flash "
        f"(eigener Kaggle-Lauf), Mischgewicht W = {W}. Alle Credits unten erhalten."})
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    OUT.mkdir(exist_ok=True)
    (OUT / "v6_hgb.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    meta.update(id="linuszarse/ntl-701-fork-v6-hgb", title="NTL 701 Fork v6 hgb", code_file="v6_hgb.ipynb",
                is_private=True, dataset_sources=meta["dataset_sources"] + [DATASET])
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    size = (OUT / "v6_hgb.ipynb").stat().st_size
    assert size < 1_000_000, f"Notebook {size} Bytes >= 1 MB (Kaggle-Limit)"
    print(f"geschrieben: {OUT}/v6_hgb.ipynb ({size / 2**10:.0f} KB), W={W}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--pool-weight", type=float, default=0.5, help="nur Beschriftung; Daten liegen im Datensatz")
    a = ap.parse_args()
    build(a.W, a.pool_weight)
