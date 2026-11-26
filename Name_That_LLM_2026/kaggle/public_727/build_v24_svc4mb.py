"""Erzeugt kaggle/public_727/v24{a,b}_svc4/ – V23 (lightningv08 0.727 unverändert) mit neu gefitteter svc4.

Experiment 18: svc4 (LinearSVC des vorab trainierten Kerns `ntl666-prefit`, Blendgewicht 12) wird mit denselben
Vektorisierern neu gefittet (Kernel `linuszarse/ntl-svc4-markbarney`, Kaggle CPU), einmal ohne (a, Kontrolle:
nur Pool-Version neu) und einmal mit den host-ähnlichen markbarney-Texten (b). OOF-Test mit Pool: +1,07 Pp.
[+0,44; +1,81] im Einzelleser (Sinkhorn).

Änderungen, sonst nichts:
- neue Zelle nach Zelle 1 (läuft auch im FAST_COMMIT): angehängte svc4-Gewichte finden und prüfen.
- Zelle 5: nach dem Entpacken des Kerns `core/svc4/model.npz` durch unsere Gewichte ersetzen.
- kernel-metadata.json: Kernel-Quelle linuszarse/ntl-svc4-markbarney, neue ID/Titel.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_727/build_v24_svc4mb.py
    .venv/bin/kaggle kernels push -p kaggle/public_727/v24b_svc4      (bzw. v24a_svc4)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v23_fork" / "v23_fork.ipynb"
SOURCE_KERNEL = "linuszarse/ntl-svc4-markbarney"
PATCH_MARK = "    for name in ('core/svc4', 'core/tfidf', 'core/distort'):\n        assert os.path.isdir(f'{PREFIT}/{name}'), name\n"
VARIANTS = {"a": ("ohne", "svc4 refit without markbarney (control: newer Open Pool only)"),
            "b": ("mit", "svc4 refit with markbarney host-style passages (w0.5)")}

CHECK_CELL = '''# ours (V24{v}): locate and check the refitted svc4 weights (runs in the commit run as well)
import glob as _g, numpy as _np
_OURS_SVC4 = sorted(_g.glob('/kaggle/input/**/svc4_{tag}_model.npz', recursive=True))
assert len(_OURS_SVC4) == 1, _OURS_SVC4
_OURS_SVC4 = _OURS_SVC4[0]
with _np.load(_OURS_SVC4) as _z:
    assert _z['coef'].shape == (18, 540000) and _z['intercept'].shape == (18,), {{k: _z[k].shape for k in _z.files}}
    assert _z['classes'].tolist() == list(range(18))
    assert _np.isfinite(_z['coef']).all() and _np.isfinite(_z['intercept']).all()
print('V24{v} svc4 weights ok:', _OURS_SVC4, flush=True)
'''

REPLACE = '''    import shutil as _sh   # ours (V24{v}): replace svc4 weights of the prefit core
    _sh.copyfile(_OURS_SVC4, f'{{PREFIT}}/core/svc4/model.npz')
    print('V24{v} svc4 model.npz replaced from', _OURS_SVC4, flush=True)
'''


def build(v):
    tag, desc = VARIANTS[v]
    nb = json.loads(SRC.read_text())
    cells = nb["cells"]
    hit = [i for i, c in enumerate(cells) if PATCH_MARK in "".join(c["source"])]
    assert hit == [5], hit
    s = "".join(cells[5]["source"])
    assert s.count(PATCH_MARK) == 1
    cells[5]["source"] = s.replace(PATCH_MARK, PATCH_MARK + REPLACE.format(v=v))
    assert "FAST_COMMIT = " in "".join(cells[1]["source"])
    cells.insert(2, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": CHECK_CELL.format(v=v, tag=tag)})
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"**V24{v} (private fork):** lightningv08's 0.727 notebook, unchanged except that the svc4 member of "
                     f"the prefit core is refitted with the same vectorizers ({desc}; kernel `{SOURCE_KERNEL}`). "
                     "All credit for the pipeline goes to the authors below."})
    out = HERE / f"v24{v}_svc4"
    out.mkdir(exist_ok=True)
    (out / f"v24{v}_svc4.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = json.loads((HERE / "v23_fork" / "kernel-metadata.json").read_text())
    meta.update(id=f"linuszarse/ntl-727-fork-v24{v}-svc4", title=f"NTL 727 Fork v24{v} svc4", code_file=f"v24{v}_svc4.ipynb")
    if SOURCE_KERNEL not in meta["kernel_sources"]:
        meta["kernel_sources"].append(SOURCE_KERNEL)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", out)


if __name__ == "__main__":
    for v in VARIANTS:
        build(v)
