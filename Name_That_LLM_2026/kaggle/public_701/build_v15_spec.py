"""Erzeugt V15/V16: V6 (0.707) + Familien-Spezialist als zusätzliche Apex-Evidenz (Experiment 12).

Apex verfeinert innerhalb jeder Familie (Claude, GPT, Gemini) die Stack-Verteilung mit einer Geschwister-LR auf
Satzzeichen/Funktionswörtern (Stärke 0.10, nur bei knappen Entscheidungen). Neu: Die Evidenz ist LR x Spezialist^t,
wobei der Spezialist je Familie TF-IDF (Wort 1-2, Zeichen 2-5) + logistische Regression auf bis zu 15.000
Open-Pool-Texten je Klasse ist (`linuszarse/ntl-family-spec`, exportiert von experiments/12_family_specialist/
export_spec.py ohne Pickle). Simulation in der V6-Kette (Ersatz-Stack): t 1.0 / Stärke 0.25 +0,73 Pp.
[+0,22; +1,26], t 1.5 / 0.20 +0,74.

Änderungen an V6 (sonst bitgleich):
- Zelle 9 (pngy_residual.py): predict_log_probability(base, texts, extra=None, strength=0.10).
- neue Zelle vor Apex: lädt den Spezialisten, prüft ihn gegen Referenzwerte für 20 train-Texte (auch im Commit).
- Apex-Zelle: Evidenz x Spezialist^t, Stärke s; scheitert der Spezialist, verhält sich die Zelle wie V6.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v15_spec.py --name v15_spec --t 1.0 --strength 0.25
    .venv/bin/kaggle kernels push -p kaggle/public_701/v15_spec
"""
import argparse
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
BASE = SRC / "v6_hgb"
DATASET = "linuszarse/ntl-family-spec"

RES_OLD1 = "    def predict_log_probability(self, base, texts):\n"
RES_NEW1 = "    def predict_log_probability(self, base, texts, extra=None, strength=0.10):   # ours (V15): extra evidence\n"
RES_OLD2 = "        return refine(base, evidence, self.families)\n"
RES_NEW2 = ("        if extra is not None:   # ours (V15): family specialist evidence times the sibling LR\n"
            "            evidence = evidence * np.asarray(extra, dtype=float)\n"
            "        return refine(base, evidence, self.families, strength=strength)\n")
APEX_OLD = "        refined_lp = post_model.predict_log_probability(base_lp, query_df.text.fillna('').astype(str))\n"
APEX_NEW = """        try:   # ours (V15): family specialist evidence (TF-IDF + LR on the Open Pool), see the cell above
            _spec_ev = family_spec_evidence(query_df.text.fillna('').astype(str).tolist(), labels) ** SPEC_T
            print(f'V15 family specialist evidence {_spec_ev.shape}, t {SPEC_T}, strength {SPEC_STRENGTH}', flush=True)
        except Exception as _e_spec:
            _spec_ev = None
            print(f'V15 family specialist FAILED, V6 behaviour: {_e_spec!r}', flush=True)
        refined_lp = post_model.predict_log_probability(base_lp, query_df.text.fillna('').astype(str), extra=_spec_ev,
                                                        strength=SPEC_STRENGTH if _spec_ev is not None else 0.10)
"""

SPEC_CELL = """# ---- ours (V15): family specialist for the Apex sibling refinement ----
# Per family (Claude, GPT, Gemini): TF-IDF (word 1-2, char_wb 2-5) + logistic regression on up to 15,000 Open-Pool
# texts per class (crtproduction, CC BY-SA 4.0), exported without pickle (vocabulary, idf, coefficients).
# Apex then uses sibling-LR evidence x specialist^SPEC_T at strength SPEC_STRENGTH instead of the LR alone at 0.10.
import glob, json
from pathlib import Path
from scipy.sparse import hstack as _hstack
from scipy.special import softmax as _softmax
from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
SPEC_T, SPEC_STRENGTH = %(t)r, %(s)r
_SPEC_KW = {'word': dict(ngram_range=(1, 2), sublinear_tf=True),
            'char': dict(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True)}
_hits = sorted(glob.glob('/kaggle/input/**/ntl-family-spec/**/claude.npz', recursive=True)) or \\
        sorted(glob.glob('/kaggle/input/**/claude.npz', recursive=True))
assert _hits, 'V15: family specialist files not mounted'
SPEC_DIR = Path(_hits[0]).parent
_SPEC = {}

def _spec_models():
    if not _SPEC:
        for fam in ('claude', 'gpt', 'gemini'):
            z = np.load(SPEC_DIR / f'{fam}.npz', allow_pickle=False)
            vecs = []
            for kind in ('word', 'char'):
                terms = json.loads((SPEC_DIR / f'{fam}_vocab_{kind}.json').read_text())
                v = _Tfidf(vocabulary={t: i for i, t in enumerate(terms)}, **_SPEC_KW[kind])
                v.idf_ = z[f'idf_{kind}']
                vecs.append(v)
            _SPEC[fam] = (vecs, z['coef'].astype(np.float64), z['intercept'].astype(np.float64), z['classes'].astype(str))
    return _SPEC

def family_spec_evidence(texts, labels):
    ev = np.ones((len(texts), len(labels)))
    for fam, (vecs, coef, intercept, classes) in _spec_models().items():
        X = _hstack([v.transform(texts) for v in vecs]).tocsr()
        p = _softmax(X @ coef.T + intercept, axis=1)
        for j, c in enumerate(classes):
            ev[:, list(labels).index(c)] = p[:, j]
    return ev

# smoke test (commit and rerun): reference values for 20 train texts computed locally with the fitting sklearn
_ref = np.load(SPEC_DIR / 'reference.npz', allow_pickle=False)
_tr = pd.read_csv(os.path.join(DATA, 'train.csv')).astype({'id': str}).set_index('id')
_labels = sorted(_tr.model.unique())
_ev = family_spec_evidence(_tr.loc[_ref['ids'].astype(str), 'text'].fillna('').astype(str).tolist(), _labels)
_err = float(np.abs(_ev - _ref['evidence']).max())
assert _err < 1e-4, f'V15: specialist differs from the local reference by {_err:.2e} (sklearn version?)'
_te = pd.read_csv(os.path.join(DATA, 'test.csv'))
_evt = family_spec_evidence(_te.text.fillna('').astype(str).tolist()[:50], _labels)
assert np.isfinite(_evt).all() and _evt.shape == (min(50, len(_te)), 18)
print(f'V15 family specialist ok: {SPEC_DIR}, max |diff| to reference {_err:.1e}, t {SPEC_T}, strength {SPEC_STRENGTH}', flush=True)
"""


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def build(name, t, s):
    nb = json.loads((BASE / "v6_hgb.ipynb").read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    cells = nb["cells"]
    res = [i for i, c in enumerate(cells) if src(c).startswith("%%writefile pngy_residual.py")]
    apx = [i for i, c in enumerate(cells) if "APEX GRANDMASTER V2" in src(c)]
    assert len(res) == 1 and len(apx) == 1, (res, apx)
    r = src(cells[res[0]])
    assert r.count(RES_OLD1) == 1 and r.count(RES_OLD2) == 1
    cells[res[0]]["source"] = r.replace(RES_OLD1, RES_NEW1).replace(RES_OLD2, RES_NEW2)
    compile(src(cells[res[0]]).split("\n", 1)[1], "pngy", "exec")
    x = src(cells[apx[0]])
    assert x.count(APEX_OLD) == 1
    cells[apx[0]]["source"] = x.replace(APEX_OLD, APEX_NEW)
    compile(src(cells[apx[0]]), "apex", "exec")
    spec = SPEC_CELL % {"t": t, "s": s}
    compile(spec, "spec", "exec")
    cells.insert(apx[0], {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": spec})
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    assert src(cells[0]).startswith("# Fork V6")
    cells[0]["source"] = (
        f"# Fork {name.split('_')[0].upper()}: 0.701 + GLM likelihood classifier + family specialist in Apex\n\n"
        "Like V6 (public 0.701 by alexchilton + our HGB GLM likelihood classifier). New: the Apex sibling refinement "
        "uses its sibling-LR evidence times a family specialist (TF-IDF + logistic regression per family on up to "
        f"15,000 Open-Pool texts per class), exponent t = {t}, strength {s} instead of 0.10. All credits below kept.")
    out = SRC / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    slug = f"ntl-701-fork-{name.replace('_', '-')}"
    meta.update(id=f"linuszarse/{slug}", title=f"NTL 701 Fork {name.replace('_', ' ')}", code_file=f"{name}.ipynb",
                is_private=True)
    if DATASET not in meta["dataset_sources"]:
        meta["dataset_sources"].append(DATASET)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = (out / f"{name}.ipynb").stat().st_size
    assert size < 1_000_000, size
    print(f"geschrieben: {out}/{name}.ipynb ({size / 2**10:.0f} KB) | {slug} | t {t}, Stärke {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=0.25)
    a = ap.parse_args()
    build(a.name, a.t, a.strength)
