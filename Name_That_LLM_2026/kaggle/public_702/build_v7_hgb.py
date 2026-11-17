"""Erzeugt kaggle/public_702/v7_hgb/ – öffentliches 0.702-Notebook + unser GLM-Likelihood-Klassifikator.

Basis: lightningv08 „0.702 | Soft Typography + GLM + Qwen3.8-27B LoRA“ (codegleich mit alexchilton
„Name That LLM 0.702: All Public + What Failed“), unverändert in public_702.ipynb.
Einzige Änderung wie in V5/V6: Die GLM-Stufe nutzt statt der LogReg (nur train.csv) ein HistGradientBoosting
auf train.csv + 5.670 Open-Pool-Texten (Experiment 08). Das Modell wird einmal im Notebook aus den Merkmalen
in linuszarse/ntl-glm-ll-train trainiert (kein Pickle im Datensatz -> keine sklearn-Versionsprobleme) und
`member_logp` in v1_stage.py gepatcht. Auswahl (15 %, 90.000 Tokens), Temperatur, Abbruchgrenze und die
Apex-Nachbearbeitung bleiben unverändert.

V8 (--apex-post-glm): zusätzlich rechnet die Apex-Nachbearbeitung auf den Logits *nach* der GLM-Stufe
(post-glm.npz) statt auf den Stack-Logits davor. Sonst überschreibt Apex jede Nicht-GLM-Vorhersage mit einem
Ergebnis, das die GLM-Stufe nie gesehen hat (Experiment 09: +0,4 Pp. in der Simulation auf train.csv).
GLM-Labels bleiben wie im Original erhalten (corrected=False).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_702/build_v7_hgb.py --W 1.0                 # -> v7_hgb/
    .venv/bin/python kaggle/public_702/build_v7_hgb.py --W 1.0 --apex-post-glm # -> v8_postapex/
    .venv/bin/kaggle kernels push -p kaggle/public_702/v7_hgb
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "public_702"
DATASET = "linuszarse/ntl-glm-ll-train"
HGB_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                  min_samples_leaf=40, early_stopping=False)
POOL_WEIGHT = 0.5   # steckt bereits in den Gewichten w von ll_train.npz (build_v5_hgb.py --pool-weight 0.5)

PATCH = '''# ours (V7): GLM likelihood classifier = gradient boosting on train.csv + 5,670 Open-Pool texts scored under
# GLM-5.3 and GLM-5.3-Flash (our own Kaggle run, features in the private dataset linuszarse/ntl-glm-ll-train).
# Patches v1_stage.py (member_logp) and ll_model.json; selection, temperature, the max-changed guard and the
# Apex post-processing are unchanged. Fitted once here, before the stage runs.
import json, pickle, time
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
_t = time.time()
_OLD = __OLD__
_NEW = __NEW__
_stage = V1_DIR / 'v1_stage.py'
_src = _stage.read_text()
assert _src.count(_OLD) == 1, 'V7: member_logp in v1_stage.py is not the expected one'
_stage.write_text(_src.replace(_OLD, _NEW))
_hits = sorted({p.resolve() for g in ('ntl-glm-ll-train/ll_train.npz', '*/linuszarse/ntl-glm-ll-train/ll_train.npz')
                for p in Path('/kaggle/input').glob(g)})
assert len(_hits) == 1, f'V7: ll_train.npz not found exactly once: {_hits}'
_m = json.loads((V1_DIR / 'll_model.json').read_text())
with np.load(_hits[0], allow_pickle=False) as _d:
    _Xtr, _ytr, _wtr = _d['X'], _d['y'].astype(str), _d['w']
assert _Xtr.shape == (len(_ytr), len(_m['features'])) and sorted(set(_ytr)) == sorted(_m['classes'])
_clf = HistGradientBoostingClassifier(random_state=0, **__PARAMS__).fit(_Xtr, _ytr, sample_weight=_wtr)
with open(V1_DIR / 'll_hgb.pkl', 'wb') as _f:
    pickle.dump(_clf, _f)
_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W=__W__,
          source='ours: HGB train+pool, pool_weight __PW__, ' + str(len(_ytr)) + ' rows')
(V1_DIR / 'll_model.json').write_text(json.dumps(_m))
print('V7 LL model:', _m['kind'], 'W', _m['W'], _m['source'], 'fit', round(time.time() - _t), 's', flush=True)
if FAST_COMMIT:   # commit only: the patched scoring code path on 5 train rows
    import importlib, sys
    if str(V1_DIR) not in sys.path:
        sys.path.insert(0, str(V1_DIR))
    import v1_stage; importlib.reload(v1_stage)
    _X5 = _Xtr[:5].astype(np.float64)
    _feats = {t: pd.DataFrame({f.split(':', 1)[1]: _X5[:, j] for j, f in enumerate(_m['features']) if f.startswith(t + ':')},
                              index=list(range(5))) for t in ('flash', 'glm53')}
    _lp = v1_stage.member_logp(_m, _feats, list(range(5)))
    assert _lp.shape == (5, 18) and np.isfinite(_lp).all() and np.allclose(np.exp(_lp).sum(1), 1, atol=1e-6)
    print('V7 member_logp smoke OK, train labels', list(_ytr[:5]), '-> argmax',
          [_m['classes'][i] for i in _lp.argmax(1)], flush=True)
'''

OLD_Z = ("    z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T "
         "+ np.asarray(model['intercept'])\n    return log_softmax(z, 1)")
NEW_Z = ("    if model.get('kind') == 'hgb':   # ours (V7): boosted trees on train + Open-Pool features, fitted in the notebook\n"
         "        import pickle\n"
         "        with open(model['clf_file'], 'rb') as f:\n"
         "            clf = pickle.load(f)\n"
         "        p = clf.predict_proba(X)[:, [list(clf.classes_).index(c) for c in model['classes']]]\n"
         "        return np.log(np.clip(p, 1e-9, 1))\n" + OLD_Z)


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


APEX_OLD = "final=apply_postprocessing(train_df,query_df,FINAL_LOGITS,str(V1_OUT/'post-glm.npz'),OUT,"
APEX_NEW = "final=apply_postprocessing(train_df,query_df,str(V1_OUT/'post-glm.npz'),str(V1_OUT/'post-glm.npz'),OUT,"


def build(W, apex_post_glm=False):
    tag, name = ("v8_postapex", "V8") if apex_post_glm else ("v7_hgb", "V7")
    OUT = SRC / tag
    nb = json.loads((SRC / "public_702.ipynb").read_text())
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    cells = [src(c) for c in nb["cells"]]

    def one(pred, what):
        hits = [i for i, s in enumerate(cells) if pred(s)]
        assert len(hits) == 1, f"{what}: {len(hits)} Treffer – Basis-Notebook hat sich geändert"
        return hits[0]

    i_stage = one(lambda s: s.startswith("%%writefile /kaggle/temp/v1/v1_stage.py"), "v1_stage.py")
    i_model = one(lambda s: "ll_model.json').write_text" in s, "ll_model.json")
    i_dir = one(lambda s: "V1_DIR = Path('/kaggle/temp/v1')" in s, "V1_DIR")
    i_run = one(lambda s: "str(V1_DIR/'v1_stage.py')" in s and "subprocess.run(command" in s, "Stufe starten")
    assert cells[i_stage].count(OLD_Z) == 1, "member_logp in v1_stage.py hat sich geändert"
    assert i_stage < i_model < i_dir < i_run, (i_stage, i_model, i_dir, i_run)

    code = (PATCH.replace("__OLD__", repr(OLD_Z)).replace("__NEW__", repr(NEW_Z))
            .replace("__PARAMS__", repr(HGB_PARAMS)).replace("__W__", repr(W)).replace("__PW__", str(POOL_WEIGHT)))
    compile(code, "patch", "exec")
    nb["cells"].insert(i_dir + 1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                   "source": code})
    if apex_post_glm:
        i_apex = one(lambda s: APEX_OLD in s, "Apex-Aufruf")
        cell = nb["cells"][i_apex + 1]   # +1: Patch-Zelle davor eingefügt
        assert src(cell).count(APEX_OLD) == 1
        cell["source"] = ("# ours (V8): Apex refines the logits after the GLM stage (post-glm.npz), not the stack logits\n"
                          "# before it, so it no longer overwrites non-GLM changes of the GLM stage; GLM labels are kept\n"
                          + src(cell).replace(APEX_OLD, APEX_NEW))
    extra = (" Zusätzlich (V8): Die Apex-Nachbearbeitung rechnet auf den Logits nach der GLM-Stufe statt davor."
             if apex_post_glm else "")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source":
        f"# Fork {name}: 0.702 + GLM-Likelihood-Klassifikator (Gradient Boosting mit Open-Pool-Daten)\n\n"
        "Unverändert: lightningv08 *0.702 | Soft Typography + GLM + Qwen3.8-27B LoRA* (codegleich mit alexchilton "
        f"*Name That LLM 0.702: All Public + What Failed*). {'Unsere Änderungen' if apex_post_glm else 'Unsere einzige Änderung'}: "
        "Die GLM-Stufe (José Padilla) nutzt "
        "statt der LogReg auf train.csv ein Gradient Boosting auf train.csv + 5.670 Open-Pool-Texten, bewertet unter "
        f"GLM-5.3 und GLM-5.3-Flash (eigener Kaggle-Lauf), Mischgewicht W = {W}.{extra} Alle Credits unten erhalten."})
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    slug = tag.replace("_", "-")
    meta.update(id=f"linuszarse/ntl-702-fork-{slug}", title=f"NTL 702 Fork {slug.replace('-', ' ')}",
                code_file=f"{tag}.ipynb", is_private=True, dataset_sources=meta["dataset_sources"] + [DATASET])
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"geschrieben: {OUT}/{tag}.ipynb ({(OUT / f'{tag}.ipynb').stat().st_size / 2**20:.2f} MB), "
          f"Patch-Zelle nach Zelle {i_dir} (V1_DIR), W={W}, Apex nach GLM-Stufe: {apex_post_glm}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--apex-post-glm", action="store_true", help="V8: Apex auf den Logits nach der GLM-Stufe")
    a = ap.parse_args()
    build(a.W, a.apex_post_glm)
