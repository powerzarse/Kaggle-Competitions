"""Erzeugt V18/V19: V6 (0.707), GLM-Stufe liest die ersten 256 statt 160 Tokens (Experiment 14).

Anlass: omerfarukmerey/0-706-glm-fable-routing-256-token-likelihood (Public 0.706) ändert gegenüber lightningv08s 0.704
nur den Schnitt der GLM-Stufe (160 → 256 Tokens, Kopf mit estopadillas Rezept neu angepasst): +0,002 auf dem LB.
Kopf allein (OOF, 18 Klassen): 0.190 bei 160 → 0.215 bei 256 Tokens.

- v18  Kopf = öffentlicher LR-Kopf bei 256 Tokens (aus dem 0.706-Notebook, `experiments/14_glm_256/
       ll_model_L256_public.json`). Unser HGB fällt weg: reine Übertragung der öffentlichen Änderung.
- v19  Kopf = Mischung im Log-Raum: a · HGB-160 (V6, train + Open Pool) + (1 − a) · LR-256. Die 160er-Merkmale für
       den HGB werden aus den Token-Scores desselben 256er-Laufs nachgerechnet (`features.one(s[:160], n_chars)`,
       kausales Modell, exakt). Prüfung zur Laufzeit: derselbe Code auf dem vollen Schnitt reproduziert die
       Merkmale der Pipeline, sonst Abbruch. Im Commit läuft der Mischpfad auf synthetischen Token-Scores.

Auswahl, Deadline (8,8 h), Temperatur, Abbruchgrenze und Apex bleiben wie in V6. Mit 256 Tokens kostet jeder Text
mehr Zeit, die Abdeckung der Stufe sinkt von ~20 % auf ~17 %.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v18_l256.py --variant v18
    .venv/bin/python kaggle/public_701/build_v18_l256.py --variant v19 --mix 0.5
    .venv/bin/kaggle kernels push -p kaggle/public_701/v18_l256      # bzw. v19_mix
"""
import argparse
import ast
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
BASE = SRC / "v6_hgb"
HEAD = "# Fork V6"
LL256 = ROOT / "experiments" / "14_glm_256" / "ll_model_L256_public.json"
PATCH_HEAD = "# ours (V6): GLM likelihood classifier"

# --- V18: ersetzt die ganze Patch-Zelle von V6 ---
V18_CELL = '''# ours (V18): the GLM stage reads the first 256 tokens instead of 160. Member = estopadilla's likelihood head
# (54 features, StandardScaler + multinomial LR, C 0.05) refitted at 256 tokens, as published in omerfarukmerey's
# 0.706 notebook (embedded verbatim below). v1_stage.py is unchanged: it passes the head's max_text_tokens (256) to the
# GLM runner and asserts the cut. Selection, deadline, temperature and the max-changed guard are as in V6.
import json
import numpy as np, pandas as pd
_LL256 = %(ll)r
_m = json.loads(_LL256)
_old = json.loads((V1_DIR / 'll_model.json').read_text())
assert _m['format'] == _old['format'] and _m['features'] == _old['features'] and _m['classes'] == _old['classes']
assert _m['max_text_tokens'] == 256 and _old['max_text_tokens'] == 160
(V1_DIR / 'll_model.json').write_text(json.dumps(_m))
print('V18 LL model: LR at', _m['max_text_tokens'], 'tokens, W', _m['W'], '|', _m['source'], flush=True)
if FAST_COMMIT:   # commit only: the unchanged scoring code path on random features
    import importlib, sys
    sys.path.insert(0, str(V1_DIR))
    import v1_stage; importlib.reload(v1_stage)
    _rng = np.random.default_rng(0)
    _cols = [f.split(':', 1)[1] for f in _m['features'] if f.startswith('flash:')]
    _feats = {t: pd.DataFrame(np.asarray(_m['mean'])[[_m['features'].index(f'{t}:{c}') for c in _cols]]
                              + _rng.normal(size=(5, len(_cols))), columns=_cols, index=list(range(5)))
              for t in ('flash', 'glm53')}
    _lp = v1_stage.member_logp(_m, _feats, list(range(5)))
    assert _lp.shape == (5, 18) and np.isfinite(_lp).all() and np.allclose(np.exp(_lp).sum(1), 1, atol=1e-6)
    print('V18 member_logp smoke OK', flush=True)
'''

# --- V19: Änderungen an der Patch-Zelle von V6 ---
RET_HGB = "        return np.log(np.clip(p, 1e-9, 1))\n"
MIX_CODE = '''    if model.get('kind') == 'mix':   # ours (V19): log-mix of the HGB at 160 tokens and the LR at 256 tokens
        import pickle
        import features as _features
        z = ((X - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coef']).T + np.asarray(model['intercept'])
        lr = log_softmax(z, 1)
        cut = int(model['hgb_cut'])
        blocks, full = [], []
        for t in ('flash', 'glm53'):
            q = pd.read_csv(Path(model['work']) / f'v1_query_{t}.csv')   # the file run_pipeline.py read
            n_chars = dict(zip(q['id'].astype(str), q['text'].astype(str).str.len()))
            with np.load(Path(model['outdir']) / f'{t}_scores.npz', allow_pickle=True) as d:   # ids: object array (run_pipeline)
                sid, off, sc = d['ids'].astype(str), d['offsets'], d['scores']
            cut_rows, full_rows = {}, {}
            for k, i in enumerate(sid):
                s = sc[off[k]:off[k + 1]]
                cut_rows[i] = _features.one(s[:cut], n_chars[i])
                full_rows[i] = _features.one(s, n_chars[i])
            blocks.append(pd.DataFrame.from_dict(cut_rows, orient='index').add_prefix(f'{t}:'))
            full.append(pd.DataFrame.from_dict(full_rows, orient='index').add_prefix(f'{t}:'))
        def _matrix(parts):
            G = pd.concat(parts, axis=1).reindex(ids)
            for c in model['diff_features']:
                G[f'd:{c}'] = G[f'flash:{c}'] - G[f'glm53:{c}']
            return G[model['features']].to_numpy(np.float64)
        X_full, X_cut = _matrix(full), _matrix(blocks)
        # the same feature code on the stage's own cut must reproduce the pipeline's features (up to the CSV round
        # trip: logrank_mean stays float32 in features.one, ~1e-8 relative)
        assert np.allclose(X_full, X, rtol=1e-5, atol=1e-6), f'V19: recomputed features differ (max {np.abs(X_full - X).max()})'
        assert np.isfinite(X_cut).all(), 'V19: non-finite 160-token features'
        with open(model['clf_file'], 'rb') as f:
            clf = pickle.load(f)
        p = clf.predict_proba(X_cut)[:, [list(clf.classes_).index(c) for c in model['classes']]]
        hgb = np.log(np.clip(p, 1e-9, 1))
        a = float(model['mix_hgb'])
        return log_softmax(a * hgb + (1 - a) * lr, 1)
'''
M_OLD = "_m = json.loads((V1_DIR / 'll_model.json').read_text())\n"
M_NEW = ("_m = json.loads((V1_DIR / 'll_model.json').read_text())   # V6's 160-token head: features and classes for the HGB\n"
         "_LL256 = %(ll)r   # ours (V19): estopadilla's LR recipe at 256 tokens (omerfarukmerey's 0.706 head)\n"
         "_m256 = json.loads(_LL256)\n"
         "assert _m256['features'] == _m['features'] and _m256['classes'] == _m['classes'] and _m['max_text_tokens'] == 160\n")
UPD_OLD = ("_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W=1.0,\n"
           "          source='ours: HGB train+pool, pool_weight 0.5, ' + str(len(_ytr)) + ' rows')\n")
UPD_NEW = ("_m = dict(_m256, kind='mix', clf_file=str(V1_DIR / 'll_hgb.pkl'), W=1.0, hgb_cut=160, mix_hgb=%(mix)r,\n"
           "          outdir=str(V1_OUT), work='/kaggle/temp/v1-work',   # as passed to v1_stage.py by the stage cell\n"
           "          source='ours (V19): %(mix)r x HGB-160 (train+pool, pool_weight 0.5, ' + str(len(_ytr)) + ' rows) + LR-256')\n")
PRINT_OLD = "print('V6 LL model:', _m['kind'], 'W', _m['W'], _m['source'], 'fit', round(time.time() - _t), 's', flush=True)\n"
PRINT_NEW = ("print('V19 LL model:', _m['kind'], 'W', _m['W'], 'cut', _m['max_text_tokens'], _m['source'], 'fit',\n"
             "      round(time.time() - _t), 's', flush=True)\n")
SMOKE_START = "if FAST_COMMIT:   # commit only: the patched scoring code path on 5 train rows\n"
SMOKE_V19 = '''if FAST_COMMIT:   # commit only (V19): the mix path on synthetic per-token scores of 5 train texts
    import importlib, sys, tempfile
    sys.path.insert(0, str(V1_DIR))
    import v1_stage, features as _features; importlib.reload(v1_stage)
    _tmp = Path(tempfile.mkdtemp())
    _tr5 = pd.read_csv(os.path.join(DATA, 'train.csv')).head(5)[['id', 'text']]
    _rng = np.random.default_rng(0)
    for _t in ('flash', 'glm53'):
        _tr5.to_csv(_tmp / f'v1_query_{_t}.csv', index=False)
        _q = pd.read_csv(_tmp / f'v1_query_{_t}.csv')
        _sc = []
        for _n in _rng.integers(120, 257, size=5):
            _h = _rng.uniform(0.2, 5, _n)
            _sc.append(np.c_[-_rng.exponential(2.5, _n), _h, _h * _h + _rng.uniform(0.5, 4, _n),
                             np.floor(_rng.exponential(30, _n))].astype(np.float32))
        _off = np.r_[0, np.cumsum([len(s) for s in _sc])]
        np.savez_compressed(_tmp / f'{_t}_scores.npz', ids=_q['id'].astype(str).values, scores=np.concatenate(_sc), offsets=_off)
        _f = _features.text_features(_sc, _q['text'].astype(str).tolist())
        _f.insert(0, 'id', _q['id'].astype(str).values)
        _f.to_csv(_tmp / f'{_t}_features.csv', index=False)
    _sel = _tr5['id'].astype(str).tolist()
    _feats = {t: pd.read_csv(_tmp / f'{t}_features.csv', dtype={'id': str}).set_index('id').loc[_sel] for t in ('flash', 'glm53')}
    _lp = v1_stage.member_logp(dict(_m, outdir=str(_tmp), work=str(_tmp)), _feats, _sel)
    assert _lp.shape == (5, 18) and np.isfinite(_lp).all() and np.allclose(np.exp(_lp).sum(1), 1, atol=1e-6)
    print('V19 member_logp mix smoke OK, argmax', [_m['classes'][i] for i in _lp.argmax(1)], flush=True)
'''


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def patch_v19(s, ll, mix):
    line = [l for l in s.split("\n") if l.startswith("_NEW = ")]
    assert len(line) == 1
    new_code = ast.literal_eval(line[0][len("_NEW = "):])
    assert new_code.count(RET_HGB) == 1 and new_code.startswith("    if model.get('kind') == 'hgb':")
    s = s.replace(line[0], "_NEW = " + repr(MIX_CODE + new_code))
    for old, new in ((M_OLD, M_NEW % {"ll": ll}), (UPD_OLD, UPD_NEW % {"mix": mix}), (PRINT_OLD, PRINT_NEW)):
        assert s.count(old) == 1, f"nicht eindeutig: {old[:80]!r}"
        s = s.replace(old, new)
    i = s.index(SMOKE_START)
    return s[:i] + SMOKE_V19


def build(variant, mix):
    ll = json.dumps(json.loads(LL256.read_text()))
    nb = json.loads((BASE / "v6_hgb.ipynb").read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    patch = [c for c in nb["cells"] if src(c).startswith(PATCH_HEAD)]
    assert len(patch) == 1
    if variant == "v18":
        out, slug = "v18_l256", "ntl-701-fork-v18-l256"
        s = V18_CELL % {"ll": ll}
        desc = ("Die GLM-Stufe liest die ersten 256 statt 160 Tokens; Kopf = öffentlicher LR-Kopf bei 256 Tokens "
                "(omerfarukmerey 0.706, estopadillas Rezept) statt unseres HGB.")
    else:
        tag = f"{round(mix * 100):03d}"
        out, slug = f"v19_mix{tag}", f"ntl-701-fork-v19-mix{tag}"
        s = patch_v19(src(patch[0]), ll, mix)
        desc = (f"Die GLM-Stufe liest die ersten 256 statt 160 Tokens; Kopf = {mix} × unser HGB bei 160 Tokens "
                f"(train + Open Pool, aus den 256er-Token-Scores nachgerechnet) + {1 - mix:.2f} × öffentlicher LR-Kopf "
                "bei 256 Tokens (omerfarukmerey 0.706).")
    compile(s, "patch", "exec")
    patch[0]["source"] = s
    assert src(nb["cells"][0]).startswith(HEAD)
    nb["cells"][0]["source"] = (
        f"# Fork {variant.upper()}: 0.701 + GLM-Stufe mit 256 Tokens\n\n"
        "Wie V6 (0.701 von alexchilton + unser GLM-Klassifikator), Apex unverändert. Einzige Änderung: "
        f"{desc} Alle Credits unten erhalten; zusätzlich omerfarukmerey (256-Token-Kopf).")
    for c in nb["cells"]:   # die V6-Basis ist von Kaggle gezogen und enthält Commit-Ausgaben
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    (SRC / out).mkdir(exist_ok=True)
    (SRC / out / f"{out}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    meta.update(id=f"linuszarse/{slug}", title=slug.replace("ntl-701-fork-", "NTL 701 Fork ").replace("-", " ", 1),
                code_file=f"{out}.ipynb", is_private=True)
    assert "linuszarse/ntl-glm-ll-train" in meta["dataset_sources"]
    (SRC / out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = (SRC / out / f"{out}.ipynb").stat().st_size
    assert size < 1_000_000, size
    print(f"geschrieben: {SRC / out}/{out}.ipynb ({size / 2**10:.0f} KB), Titel {meta['title']!r}: {desc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v18", "v19"], required=True)
    ap.add_argument("--mix", type=float, default=0.5, help="nur v19: Gewicht des HGB-160 im Log-Mix")
    a = ap.parse_args()
    build(a.variant, a.mix)
