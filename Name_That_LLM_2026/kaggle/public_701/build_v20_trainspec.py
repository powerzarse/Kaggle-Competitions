"""Erzeugt V20: V6 (0.707) + Familien-Spezialist nur auf train.csv als zusätzliche Apex-Evidenz (Experiment 15).

LB-Gegenprobe zu V15 (Spezialist auf dem Open Pool, 0.697): gleicher Eingriff (Apex-Evidenz = Geschwister-LR x
Spezialist^t), aber der Spezialist (TF-IDF Wort 1-2 + Zeichen 2-5, LR C 4, je Familie) wird wie die Geschwister-LR
zur Laufzeit auf train.csv gefittet. Stärke so gewählt, dass V20 in der Kette etwa so viele Vorhersagen ändert wie
V15 (112 gegen 118, experiments/15_train_spec/oof_train_spec.py). Kein Datensatz nötig.

Änderungen an V6 (sonst bitgleich), Zelle 9 und Apex-Zelle wie V15 (build_v15_spec.py):
- Zelle 9 (pngy_residual.py): predict_log_probability(base, texts, extra=None, strength=0.10).
- neue Zelle vor Apex: fittet den Spezialisten auf train.csv, prüft ihn gegen lokale Referenzwerte (auch im Commit).
- Apex-Zelle: Evidenz x Spezialist^t, Stärke s; scheitert der Spezialist, verhält sich die Zelle wie V6.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v20_trainspec.py
    .venv/bin/kaggle kernels push -p kaggle/public_701/v20_trainspec
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from build_v15_spec import APEX_NEW as _APEX_V15, APEX_OLD, BASE, RES_NEW1, RES_NEW2, RES_OLD1, RES_OLD2, src

NAME, T, STRENGTH, C = "v20_trainspec", 1.0, 0.35, 4.0
APEX_NEW = (_APEX_V15.replace("(TF-IDF + LR on the Open Pool)", "(TF-IDF + LR fitted on train.csv)").replace("ours (V15)", "ours (V20)")
            .replace("V15 family", "V20 family"))
N_REF = 20

SPEC_CELL = """# ---- ours (V20): family specialist for the Apex sibling refinement, fitted on train.csv ----
# Per family (Claude, GPT, Gemini): TF-IDF (word 1-2, char_wb 2-5, min_df 2) + logistic regression (C %(C)r),
# fitted here on train.csv like the Apex sibling LR (no external data). Apex then uses sibling-LR evidence x
# specialist^SPEC_T at strength SPEC_STRENGTH instead of the LR alone at 0.10.
import time as _time
from scipy.sparse import hstack as _hstack
from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
from sklearn.linear_model import LogisticRegression as _LR
SPEC_T, SPEC_STRENGTH = %(t)r, %(s)r
_SPEC = {}

def _spec_models():
    if not _SPEC:
        _t0 = _time.time()
        tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
        x, yy = tr.text.fillna('').astype(str).to_numpy(dtype=object), tr.model.astype(str).to_numpy()
        for fam in ('claude', 'gpt', 'gemini'):
            m = np.char.startswith(yy.astype(str), fam + '-')
            vecs = [_Tfidf(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                    _Tfidf(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)]
            X = _hstack([v.fit_transform(x[m]) for v in vecs]).tocsr()
            _SPEC[fam] = (vecs, _LR(C=%(C)r, max_iter=2000, class_weight='balanced').fit(X, yy[m]))
        print(f'V20 family specialist fitted on {len(tr)} train texts in {_time.time() - _t0:.1f} s', flush=True)
    return _SPEC

def family_spec_evidence(texts, labels):
    ev = np.ones((len(texts), len(labels)))
    for fam, (vecs, clf) in _spec_models().items():
        p = clf.predict_proba(_hstack([v.transform(texts) for v in vecs]).tocsr())
        for j, c in enumerate(clf.classes_):
            ev[:, list(labels).index(c)] = p[:, j]
    return ev

# smoke test (commit and rerun): reference values for %(n)d train texts computed locally (sklearn %(skv)s)
_REF_IDS = %(ids)r
_REF_EV = np.array(%(ref)r)
_tr = pd.read_csv(os.path.join(DATA, 'train.csv')).astype({'id': str}).set_index('id')
_labels = sorted(_tr.model.unique())
_ev = family_spec_evidence(_tr.loc[_REF_IDS, 'text'].fillna('').astype(str).tolist(), _labels)
_err = float(np.abs(_ev - _REF_EV).max())
assert _err < 0.02, f'V20: specialist differs from the local reference by {_err:.2e}'
_te = pd.read_csv(os.path.join(DATA, 'test.csv'))
_evt = family_spec_evidence(_te.text.fillna('').astype(str).tolist()[:50], _labels)
assert np.isfinite(_evt).all() and _evt.shape == (min(50, len(_te)), 18)
print(f'V20 family specialist ok, max |diff| to reference {_err:.1e}, t {SPEC_T}, strength {SPEC_STRENGTH}', flush=True)
"""


def reference():
    """Spezialist lokal genau wie in der Zelle fitten, Evidenz für N_REF train-Texte (Referenz für den Smoke-Test)."""
    import pandas as pd
    import sklearn
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from src.data import load_train
    tr = load_train()
    labels = sorted(tr.model.unique())
    x, yy = tr.text.fillna("").astype(str).to_numpy(dtype=object), tr.model.astype(str).to_numpy()
    rng = np.random.default_rng(20)
    pick = np.sort(rng.choice(len(tr), N_REF, replace=False))
    ev = np.ones((N_REF, len(labels)))
    for fam in ("claude", "gpt", "gemini"):
        m = np.char.startswith(yy.astype(str), fam + "-")
        vecs = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)]
        X = hstack([v.fit_transform(x[m]) for v in vecs]).tocsr()
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, yy[m])
        p = clf.predict_proba(hstack([v.transform(x[pick]) for v in vecs]).tocsr())
        for j, c in enumerate(clf.classes_):
            ev[:, labels.index(c)] = p[:, j]
    ids = tr.id.astype(str).to_numpy()[pick].tolist()
    return ids, np.round(ev, 6).tolist(), sklearn.__version__


def build():
    nb = json.loads((BASE / "v6_hgb.ipynb").read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    cells = nb["cells"]
    res = [i for i, c in enumerate(cells) if src(c).startswith("%%writefile pngy_residual.py")]
    apx = [i for i, c in enumerate(cells) if "APEX GRANDMASTER V2" in src(c)]
    assert len(res) == 1 and len(apx) == 1, (res, apx)
    r = src(cells[res[0]])
    assert r.count(RES_OLD1) == 1 and r.count(RES_OLD2) == 1
    cells[res[0]]["source"] = (r.replace(RES_OLD1, RES_NEW1.replace("(V15)", "(V15/V20)"))
                               .replace(RES_OLD2, RES_NEW2.replace("(V15)", "(V15/V20)")))
    compile(src(cells[res[0]]).split("\n", 1)[1], "pngy", "exec")
    x = src(cells[apx[0]])
    assert x.count(APEX_OLD) == 1
    cells[apx[0]]["source"] = x.replace(APEX_OLD, APEX_NEW)
    compile(src(cells[apx[0]]), "apex", "exec")
    ids, ref, skv = reference()
    spec = SPEC_CELL % {"t": T, "s": STRENGTH, "C": C, "n": N_REF, "ids": ids, "ref": ref, "skv": skv}
    compile(spec, "spec", "exec")
    cells.insert(apx[0], {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": spec})
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    assert src(cells[0]).startswith("# Fork V6")
    cells[0]["source"] = (
        "# Fork V20: 0.701 + GLM likelihood classifier + train.csv family specialist in Apex\n\n"
        "Like V6 (public 0.701 by alexchilton + our HGB GLM likelihood classifier). New: the Apex sibling refinement "
        "uses its sibling-LR evidence times a family specialist (TF-IDF + logistic regression per family, fitted on "
        f"train.csv only), exponent t = {T}, strength {STRENGTH} instead of 0.10. All credits below kept.")
    out = SRC / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    slug = f"ntl-701-fork-{NAME.replace('_', '-')}"
    meta.update(id=f"linuszarse/{slug}", title=f"NTL 701 Fork {NAME.replace('_', ' ')}", code_file=f"{NAME}.ipynb",
                is_private=True)
    assert "linuszarse/ntl-family-spec" not in meta["dataset_sources"]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = (out / f"{NAME}.ipynb").stat().st_size
    assert size < 1_000_000, size
    print(f"geschrieben: {out}/{NAME}.ipynb ({size / 2**10:.0f} KB) | {slug} | t {T}, Stärke {STRENGTH}, C {C}")


if __name__ == "__main__":
    build()
