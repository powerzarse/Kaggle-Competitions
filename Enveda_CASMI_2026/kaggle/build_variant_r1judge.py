"""Erzeugt Varianten des 0.495-Notebooks (kaggle/network_fr_p5_0495/) mit einem Rang-1-Richter auf Basis von FR.

Baustein "Rang-1-Richter" (Linolium):
  FR (taeyangg4, gelernter Fragment-Reranker) bewertet in der FR-3-25-Zelle ohnehin jede Kandidatenliste, darf aber
  nur die Ränge 3-25 umsortieren. Der Richter nutzt dieselben Scores für eine einzige, eng begrenzte Entscheidung:
  Rang 1 und Rang 2 werden getauscht, wenn
    1. FR den Rang-2-Kandidaten um mindestens GAP Standardabweichungen (z über die ganze Liste) besser bewertet,
    2. der Rang-2-Kandidat auch FRs Favorit unter den Top 5 ist,
    3. die Pipeline keinen starken eigenen Bibliothekstreffer hat (library_max < 0.90),
    4. der externe Bibliotheksabgleich Rang 1 nicht gesetzt hat (Rang 1 vor und nach dem Overlay gleich),
    5. Rang 1 und 2 seit der FR-Zelle unverändert sind.
  Keine Kandidaten kommen hinzu oder fallen weg. Jeder Fehler stellt die Datei Byte für Byte wieder her.

Änderungen am Notebook (jeweils genau einmal per assert geprüft):
  - FR-Zelle: speichert (Liste, Scores) je Molekül in _F3_SCORES (kein zusätzlicher Rechenaufwand).
  - neue letzte Zelle: der Richter.

Aufruf (aus Enveda_CASMI_2026/):
    python kaggle/build_variant_r1judge.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "network_fr_p5_0495"

VARIANTS = {  # Ordner -> (GAP, Slug, Titel)
    "r1judge_gap150": (1.5, "casmi26-r1judge-gap150", "CASMI26 R1 Judge GAP150"),
    "r1judge_gap075": (0.75, "casmi26-r1judge-gap075", "CASMI26 R1 Judge GAP075"),
}

PATCHES = [  # (alt, neu) in der FR-Zelle
    ("            if score_store is not None:\n                score_store[m] = per\n",
     "            if score_store is not None:\n                score_store[m] = (list(lst), per)\n"),
    ("globals().pop('_f3_parent_bytes', None)\n",
     "globals().pop('_f3_parent_bytes', None)\n_F3_SCORES = {}   # Linolium: FR-Scores je Molekül für den Rang-1-Richter\n"),
    ("_f3_rerank(_f3_ids, _f3_raw, _f3_specs, _f3_ff, _f3_nets, _F3_W)",
     "_f3_rerank(_f3_ids, _f3_raw, _f3_specs, _f3_ff, _f3_nets, _F3_W, score_store=_F3_SCORES)"),
]

MARKDOWN = """## Linolium · Rang-1-Richter (FR-Gate, GAP {gap})

FR bewertet in der Zelle davor jede Liste, darf aber nur die Ränge 3-25 umsortieren. Diese Zelle nutzt dieselben Scores
für eine einzige Entscheidung: **Rang 1 und 2 tauschen**, wenn FR den Rang-2-Kandidaten um mindestens {gap} Standardabweichungen
besser bewertet und ihn auch unter den Top 5 am höchsten einstuft. Nie angefasst werden Listen mit starkem eigenen
Bibliothekstreffer (`library_max` ≥ 0.90) und Listen, deren Rang 1 der externe Bibliotheksabgleich gesetzt hat.
Keine Kandidaten kommen hinzu oder fallen weg; bei jedem Fehler bleibt die Abgabe Byte für Byte unverändert.
"""

CELL = r'''# ── Linolium · Rang-1-Richter: FR-Scores entscheiden über Rang 1 <-> 2, eng begrenzt ──
import csv as _r1_csv, json as _r1_json, os as _r1_os, time as _r1_time, traceback as _r1_tb
from collections import Counter as _R1_Counter
from pathlib import Path as _R1_Path
import numpy as _r1_np

_R1_GAP = float('__GAP__')     # Mindestabstand z(Rang 2) - z(Rang 1)
_R1_LIB_TAU = 0.90             # Listen mit eigenem Bibliothekstreffer >= 0.90 bleiben unberührt
_R1_TOPK = 5                   # Rang 2 muss FRs Favorit unter den Top 5 sein
_R1_LIMIT_S = 8.6 * 3600       # nach 8.6 h Sitzungszeit nichts mehr ändern


def _r1_z(per):
    sc = _r1_np.array(_f3_ens(per), dtype=_r1_np.float64)
    ok = _r1_np.isfinite(sc)
    z = _r1_np.full_like(sc, _r1_np.nan)
    if ok.sum() >= 3:
        v = sc[ok]; sd = float(v.std())
        z[ok] = 0.0 if sd < 1e-8 else (v - v.mean()) / sd
    return z


def _r1_decide(cur, fr_list, per, lib, pre_top1):
    """Neue Liste (oder None) und Grund. cur = aktuelle Liste, fr_list = Liste zur Zeit der FR-Zelle."""
    if len(cur) < 2 or len(fr_list) < 2 or cur[:2] != fr_list[:2]:
        return None, 'top2_changed_since_fr'
    if lib is not None and lib >= _R1_LIB_TAU:
        return None, 'library_hit'
    if pre_top1 is not None and pre_top1 != cur[0]:
        return None, 'rank1_set_by_overlay'
    z = _r1_z(per)
    if len(z) < 2 or not (_r1_np.isfinite(z[0]) and _r1_np.isfinite(z[1])):
        return None, 'no_scores'
    top = _r1_np.where(_r1_np.isfinite(z[:_R1_TOPK]), z[:_R1_TOPK], -_r1_np.inf)
    if z[1] - z[0] < _R1_GAP or int(_r1_np.argmax(top)) != 1:
        return None, 'kept'
    return [cur[1], cur[0]] + cur[2:], 'swapped'


def _r1_read_top1(path):
    if not _R1_Path(path).is_file():
        return {}
    with open(path, newline='', encoding='utf-8') as f:
        return {r['molecule_id']: r['smiles'].split(';')[0] for r in _r1_csv.DictReader(f)}


def _r1_selftest():
    per = [[0.0, 3.0, -1.0, -1.0, -1.0, 0.0]]
    new, why = _r1_decide(list('abcdef'), list('abcdef'), per, 0.5, 'a')
    assert why == 'swapped' and new == list('bacdef'), (new, why)
    assert _r1_decide(list('abcdef'), list('abcdef'), per, 0.95, 'a')[1] == 'library_hit'
    assert _r1_decide(list('abcdef'), list('abcdef'), per, 0.5, 'x')[1] == 'rank1_set_by_overlay'
    assert _r1_decide(list('bacdef'), list('abcdef'), per, 0.5, 'b')[1] == 'top2_changed_since_fr'
    assert _r1_decide(list('abcdef'), list('abcdef'), [[0.0, 0.1, -1, -1, -1, 0]], 0.5, 'a')[1] == 'kept'
    assert _r1_decide(list('abcdef'), list('abcdef'), [[0.0, 2.0, 3.0, -1, -1, 0]], 0.5, 'a')[1] == 'kept'   # Rang 3 ist Favorit
    assert _r1_decide(list('abcdef'), list('abcdef'), [[float('nan')] * 6], 0.5, 'a')[1] == 'no_scores'
    return 'passed'


_r1_audit = dict(item='Linolium Rang-1-Richter', gap=_R1_GAP, lib_tau=_R1_LIB_TAU, topk=_R1_TOPK, status='not started')
_r1_root = _R1_Path(globals().get('ROOT', '/kaggle/working'))
_r1_sub = _r1_root / 'submission.csv'
_r1_bytes = None
try:
    _r1_audit['selftest'] = _r1_selftest()
    _r1_store = globals().get('_F3_SCORES') or {}
    _r1_elapsed = _r1_time.time() - globals().get('NB_T0', _r1_time.time())
    if not globals().get('PIPELINE_DONE'):
        _r1_audit['status'] = 'fast commit: nothing to judge, submission.csv unchanged'
    elif not _r1_store:
        _r1_audit['status'] = 'no FR scores stored, submission.csv unchanged'
    elif _r1_elapsed > _R1_LIMIT_S:
        _r1_audit['status'] = f'skipped: session at {_r1_elapsed / 3600:.2f} h'
    else:
        _r1_bytes = _r1_sub.read_bytes()
        _r1_ids, _r1_raw = _f3_read(_r1_sub)
        _r1_lib = {}
        _r1_ev = _r1_root / 'frozen_candidate_evidence.parquet'
        if _r1_ev.is_file():
            import pandas as _r1_pd
            _e = _r1_pd.read_parquet(_r1_ev, columns=['molecule_id', 'library_max'])
            _r1_lib = {str(m): float(v) for m, v in _e.groupby('molecule_id', sort=False).library_max.first().items()}
        _r1_pre = _r1_read_top1(_r1_root / 'submission_pre_apex_extlib.csv')
        _r1_audit['have_library_max'] = bool(_r1_lib)
        _r1_audit['have_pre_overlay'] = bool(_r1_pre)
        _r1_new, _r1_n, _r1_moved = {}, _R1_Counter(), []
        for _m in _r1_ids:
            _cur = _r1_raw[_m].split(';')
            _r1_new[_m] = _r1_raw[_m]
            if _m not in _r1_store:
                _r1_n['not_scored_by_fr'] += 1
                continue
            _fl, _per = _r1_store[_m]
            _nl, _why = _r1_decide(_cur, _fl, _per, _r1_lib.get(_m), _r1_pre.get(_m))
            _r1_n[_why] += 1
            if _nl is not None:
                assert sorted(_nl) == sorted(_cur) and len(_nl) == len(_cur) <= 25, _m
                _r1_new[_m] = ';'.join(_nl)
                _r1_moved.append(_m)
        _r1_audit['counts'] = dict(_r1_n)
        if _r1_moved:
            _f3_write(_r1_sub, _r1_ids, _r1_new)
            _ids2, _back = _f3_read(_r1_sub)
            assert _ids2 == _r1_ids and _back == _r1_new, 'r1: written file differs'
        _r1_audit.update(status='applied', swapped=len(_r1_moved), swapped_ids=_r1_moved[:50])
except Exception as _r1_exc:
    _r1_tb.print_exc()
    _r1_audit.update(status='submission kept after an error', exception=repr(_r1_exc)[:500])
    try:
        if _r1_bytes is not None and _r1_sub.read_bytes() != _r1_bytes:
            _r1_sub.write_bytes(_r1_bytes)
            _r1_audit['restored'] = True
    except Exception as _r1_exc2:
        _r1_audit['restore_error'] = repr(_r1_exc2)[:300]
try:
    (_r1_root / 'r1judge_audit.json').write_text(_r1_json.dumps(_r1_audit, indent=1, default=str), encoding='utf-8')
except Exception:
    pass
print('[r1judge]', _r1_json.dumps({k: _r1_audit.get(k) for k in ('status', 'gap', 'swapped', 'counts', 'exception')}, default=str))
'''


def build(folder, gap, slug, title):
    nb = json.loads((SRC / "network_fr_p5_0495.ipynb").read_text())
    fr = [c for c in nb["cells"] if c["cell_type"] == "code" and "_F3_VERSION" in "".join(c["source"])]
    assert len(fr) == 1, "FR-Zelle nicht eindeutig"
    src = "".join(fr[0]["source"])
    for old, new in PATCHES:
        assert src.count(old) == 1, f"Patch-Anker {src.count(old)}x: {old[:60]!r}"
        src = src.replace(old, new)
    fr[0]["source"] = src
    assert "".join(nb["cells"][-1]["source"]).startswith("# ── H-A-130 s5: P5 tail"), "letzte Zelle ist nicht P5"
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.format(gap=gap)})
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": CELL.replace("__GAP__", str(gap))})
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"linuszarse/{slug}", title=title, code_file=f"{folder}.ipynb", is_private=True)
    out = ROOT / folder
    out.mkdir(exist_ok=True)
    (out / f"{folder}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", out)


if __name__ == "__main__":
    for folder, (gap, slug, title) in VARIANTS.items():
        build(folder, gap, slug, title)
