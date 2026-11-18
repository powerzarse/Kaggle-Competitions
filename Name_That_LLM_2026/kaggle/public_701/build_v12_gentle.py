"""Erzeugt V12–V14: V6 (0.707) mit weniger aggressiver GLM-Stufe. Basis: v6_hgb/v6_hgb.ipynb (die abgegebene V6).

Hintergrund (18.11.): Mehr Gewicht hat auf dem LB geschadet (W 1.5: −0,5 Pp., zweimal), Apex nach der GLM-Stufe
ebenfalls (−0,6, zweimal). Vermutung: Der Klassifikator ist auf dem Test überkonfident und trennt die 16
Nicht-GLM-Klassen schlecht. Daher drei Gegenproben, jede genau eine Änderung an der Patch-Zelle von V6:

- v12  W 0.7 statt 1.0. Eine Temperatur T auf dem Klassifikator wäre für die argmax dasselbe wie W/T.
- v13  „GLM gegen den Rest“: member_logp liefert für die 16 Nicht-GLM-Klassen nur noch p(Rest)/16. Der
       Klassifikator entscheidet so nur zwischen GLM-5.3, GLM-5.3-Flash und „Rest“; welche Nicht-GLM-Klasse es
       wird, entscheidet allein der Stack (und danach wie in V6 Apex).
- v14  kalibrierter Klassifikator: HGB in CalibratedClassifierCV (--calib sigmoid|isotonic, innere GroupKFold
       nach prompt_id, Pool-Zeilen je eine Gruppe), oder --calib pool025: Pool-Gewicht 0.25 statt 0.5.

Simulation: experiments/10_v6_gentle/ (V6-Kette, nur die GLM-Entscheidung ist aussagekräftig).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v12_gentle.py --variant v12
    .venv/bin/python kaggle/public_701/build_v12_gentle.py --variant v13
    .venv/bin/python kaggle/public_701/build_v12_gentle.py --variant v14 --calib pool025   # bzw. sigmoid
    .venv/bin/kaggle kernels push -p kaggle/public_701/v12_w07      # bzw. v13_glmrest, v14_pool025, v14_sigmoid
"""
import argparse
import ast
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
BASE = SRC / "v6_hgb"
HEAD = "# Fork V6"
W_OLD = "_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W=1.0,"
FIT_OLD = ("_clf = HistGradientBoostingClassifier(random_state=0, **{'learning_rate': 0.05, 'max_iter': 300, "
           "'max_leaf_nodes': 15, 'l2_regularization': 1.0, 'min_samples_leaf': 40, 'early_stopping': False})"
           ".fit(_Xtr, _ytr, sample_weight=_wtr)")
HGB_EXPR = FIT_OLD[len("_clf = "):-len(".fit(_Xtr, _ytr, sample_weight=_wtr)")]
RET_OLD = "        return np.log(np.clip(p, 1e-9, 1))\n"
RET_NEW = ("        if model.get('glm_vs_rest'):   # ours (V13): only GLM-5.3 / GLM-5.3-Flash / rest; the stack picks within the rest\n"
           "            g = np.isin(model['classes'], ['glm-5.3', 'glm-5.3-flash'])\n"
           "            p = p.copy()\n"
           "            p[:, ~g] = p[:, ~g].sum(1, keepdims=True) / (~g).sum()\n"
           + RET_OLD)
SMOKE_OLD = "    assert _lp.shape == (5, 18) and np.isfinite(_lp).all() and np.allclose(np.exp(_lp).sum(1), 1, atol=1e-6)\n"
SMOKE_V13 = SMOKE_OLD + ("    _g = np.isin(_m['classes'], ['glm-5.3', 'glm-5.3-flash'])\n"
                         "    assert np.allclose(_lp[:, ~_g], _lp[:, ~_g][:, :1]), 'V13: non-GLM classes differ'\n")


def calib_fit(method):
    if method == "pool025":
        return ("_wtr = _wtr.astype(np.float64).copy()   # ours (V14): pool weight 0.25 instead of 0.5\n"
                "_wtr[_wtr != 1.0] *= 0.5\n"
                "assert (_wtr == 1.0).sum() == 2699, 'V14: train rows are expected to have weight 1'\n" + FIT_OLD)
    return ("# ours (V14): calibrated classifier, inner GroupKFold by prompt (pool rows: one group each)\n"
            "import os\n"
            "from sklearn.calibration import CalibratedClassifierCV\n"
            "from sklearn.model_selection import GroupKFold\n"
            "_tr = pd.read_csv(os.path.join(DATA, 'train.csv'))\n"
            "assert (_ytr[:len(_tr)] == _tr.model.astype(str).to_numpy()).all(), 'V14: ll_train.npz rows do not follow train.csv'\n"
            "_grp = np.concatenate([_tr.prompt_id.astype(str).to_numpy(), np.array([f'pool{i}' for i in range(len(_ytr) - len(_tr))])])\n"
            "_cv = list(GroupKFold(5).split(_Xtr, _ytr, _grp))\n"
            # float64: the sigmoid calibrator of older sklearn (Kaggle image) rejects the float32 weights of the npz
            f"_clf = CalibratedClassifierCV({HGB_EXPR}, method={method!r}, cv=_cv).fit("
            "_Xtr.astype(np.float64), _ytr, sample_weight=_wtr.astype(np.float64))")


def variants(calib):
    return {   # name: (Ausgabeordner, Kernel-Slug, Änderungen an der Patch-Zelle, Text für Zelle 0, Quelle in ll_model.json)
        "v12": ("v12_w07", "ntl-701-fork-v12-w07",
                [(W_OLD, W_OLD.replace("W=1.0,", "W=0.7,   # ours (V12): W 1.0 -> 0.7"))],
                "Mischgewicht des Klassifikators W = 0.7 statt 1.0."),
        "v13": ("v13_glmrest", "ntl-701-fork-v13-glmrest",
                [(W_OLD, W_OLD.replace("W=1.0,", "W=1.0, glm_vs_rest=True,   # ours (V13)")), (SMOKE_OLD, SMOKE_V13)],
                "„GLM gegen den Rest“: Der Klassifikator entscheidet nur noch zwischen GLM-5.3, GLM-5.3-Flash und "
                "„Rest“ (p(Rest) gleichmäßig auf die 16 Nicht-GLM-Klassen verteilt); welche Nicht-GLM-Klasse, "
                "bestimmt allein der Stack."),
        "v14": (f"v14_{calib}", f"ntl-701-fork-v14-{calib}",
                [(FIT_OLD, calib_fit(calib))] + ([("pool_weight 0.5, ", "pool_weight 0.25, ")] if calib == "pool025" else
                                                 [("pool_weight 0.5, ", f"pool_weight 0.5, calibrated {calib}, ")]),
                "Pool-Gewicht 0.25 statt 0.5." if calib == "pool025" else
                f"Kalibrierter Klassifikator: HGB in CalibratedClassifierCV (method={calib}, innere GroupKFold nach "
                "prompt_id)."),
    }


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def build(variant, calib):
    out, slug, edits, desc = variants(calib)[variant]
    nb = json.loads((BASE / "v6_hgb.ipynb").read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    patch = [c for c in nb["cells"] if src(c).startswith("# ours (V6): GLM likelihood classifier")]
    assert len(patch) == 1
    cell = patch[0]
    s = src(cell)
    for old, new in edits:
        assert s.count(old) == 1, f"nicht eindeutig: {old[:80]!r}"
        s = s.replace(old, new)
    if variant == "v13":   # member_logp im eingebetteten Literal _NEW (wird in v1_stage.py geschrieben)
        line = [l for l in s.split("\n") if l.startswith("_NEW = ")]
        assert len(line) == 1
        new_code = ast.literal_eval(line[0][len("_NEW = "):])
        assert new_code.count(RET_OLD) == 1
        s = s.replace(line[0], "_NEW = " + repr(new_code.replace(RET_OLD, RET_NEW)))
    compile(s, "patch", "exec")
    cell["source"] = s
    assert src(nb["cells"][0]).startswith(HEAD)
    nb["cells"][0]["source"] = (
        f"# Fork {variant.upper()}: 0.701 + GLM-Likelihood-Klassifikator, weniger aggressiv\n\n"
        "Wie V6 (0.701 von alexchilton + unser GLM-Klassifikator: Gradient Boosting auf train.csv + 5.670 Open-Pool-"
        f"Texten unter GLM-5.3 / GLM-5.3-Flash), Apex unverändert. Einzige Änderung: {desc} Alle Credits unten erhalten.")
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
    ap.add_argument("--variant", choices=["v12", "v13", "v14"], required=True)
    ap.add_argument("--calib", choices=["sigmoid", "isotonic", "pool025"], default="sigmoid", help="nur v14")
    a = ap.parse_args()
    build(a.variant, a.calib)
