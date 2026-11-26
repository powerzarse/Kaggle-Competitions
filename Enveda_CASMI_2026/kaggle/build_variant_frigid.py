"""Baustein B: FRIGID-de-novo-Kandidaten auf der öffentlichen 0.509-Basis (kaggle/vt_0509/, ishankarnick).

Idee: In der "schwachen Gruppe" (kein Bibliothekstreffer) fehlt die richtige Struktur meist ganz in der Liste; dort
hilft kein Umsortieren, nur neue Kandidaten. FRIGID (coleygroup, öffentliche Gewichte kvswim/frigidckpts-casmi26)
erzeugt Strukturen direkt aus Spektrum (MIST-Fingerprint) und Summenformel.

Was der Builder einbaut (Code-Zellen der Basis bleiben unverändert):
  1. Watcher-Zelle direkt vor dem Pipeline-Lauf. Sie startet einen Hintergrundprozess auf GPU 1 (niedrige CPU-Priorität):
     - Bewertungslauf: wartet auf den Export der Pipeline (submission.csv + execution_receipt.json), leitet je Molekül die
       Summenformel aus den Top-5-Kandidaten ab (bester Massenabgleich) und lässt FRIGID bis NB_T0 + 7.5 h generieren.
     - Commit-Lauf: Funktionstest auf 2 sichtbaren Molekülen (Installation, Modelle laden, Generierung).
     Der Runner (gloriachen0816/frigidarm-runner-assets) wird nach /kaggle/working kopiert und an einer Stelle gepatcht:
     Wheels, die nicht zur Python-Version der Sitzung passen (cp312 unter 3.13), werden vor `pip install` aussortiert.
     Ohne diesen Patch scheitert die Installation unter Python 3.13 und FRIGID läuft nie.
  2. Merge-Zelle ganz am Ende (nach dem Virtual-NP-Tier). Regel wie bei heliosli/riochiang: nur Listen mit
     library_max < 0.90; erster FRIGID-Vorschlag mit Fingerprint-Ähnlichkeit >= 0.70 und <= 30 schweren Atomen, der nicht
     schon unter den ersten INSERT_AT Plätzen steht, kommt auf Rang INSERT_AT (fehlt er, fällt der letzte Eintrag weg).
     Rang 1 wird nie geändert. Jeder Fehler lässt submission.csv unverändert.
     Im Commit-Lauf: scheitert der Funktionstest, schlägt die Zelle absichtlich fehl (Status ERROR statt COMPLETE),
     damit das ohne Zugriff auf die Logs sichtbar ist.
  3. Lange Textzellen werden auf ihre Überschrift gekürzt (Kaggle lehnt Notebooks ab ~1 MB ab).

Aufruf (aus Enveda_CASMI_2026/):
    python kaggle/build_variant_frigid.py
"""
import base64
import json
from pathlib import Path

# Ersatzmodule für Pakete, die weder im Datensatz noch im Kaggle-Image liegen (werden nur angelegt, wenn sie fehlen).
# loguru: von safe/datamol nur für einfache Log-Aufrufe genutzt. ot (POT): nur für Trainings-Augmentierung importiert.
LOGURU_SHIM = '''import logging
_LV = dict(trace=5, debug=10, info=20, success=25, warning=30, error=40, critical=50, exception=40)


class _Logger:
    def __init__(self):
        self._l = logging.getLogger('loguru-shim')

    def _emit(self, lvl, msg, *a, **k):
        try:
            text = str(msg).format(*a, **k) if (a or k) else str(msg)
        except Exception:
            text = str(msg)
        self._l.log(lvl, text)

    def __getattr__(self, n):
        if n in _LV:
            return lambda msg='', *a, **k: self._emit(_LV[n], msg, *a, **k)
        if n in ('remove', 'add', 'configure', 'level', 'enable', 'disable', 'patch', 'complete'):
            return lambda *a, **k: 0
        if n in ('bind', 'opt', 'contextualize'):
            return lambda *a, **k: self
        if n == 'catch':
            return lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        raise AttributeError(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


logger = _Logger()
'''
OT_SHIM = '''def __getattr__(name):
    raise ImportError('POT (ot) is not installed; import-only stub (optimal transport is not used for generation)')
'''

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "vt_0509"
ORIGINAL = "https://www.kaggle.com/code/ishankarnick/casmi26-lb-0-509-47-7m-virtual-natural-products"
FRIGID_DATASETS = ["kvswim/frigidckpts-casmi26", "kvswim/frigiddeps-casmi26", "kvswim/frigidsrc-casmi26",
                   "gloriachen0816/frigidarm-runner-assets"]

VARIANTS = {  # Ordner -> (Einfügeposition, Slug, Titel)
    "frigid_r2": (2, "casmi26-frigid-r2", "CASMI26 FRIGID R2"),
    "frigid_r3": (3, "casmi26-frigid-r3", "CASMI26 FRIGID R3"),
}

WATCH_MD = """## Linolium · Baustein B: FRIGID-Kandidaten (Start)

Startet FRIGID im Hintergrund auf der zweiten GPU. Im Bewertungslauf wartet es auf die Kandidatenlisten der Pipeline und
erzeugt dann für jedes Molekül neue Strukturen aus Spektrum und Summenformel. Im Commit-Lauf nur ein Funktionstest auf zwei
Molekülen. Die Zusammenführung passiert in der letzten Zelle."""

WATCH_CELL = r'''# ── Linolium · Baustein B: FRIGID-Watcher (GPU 1, Hintergrund) ──
import os as _fg_os, sys as _fg_sys, subprocess as _fg_sp, time as _fg_time, json as _fg_json
from pathlib import Path as _FG_Path

_FG_WORK = _FG_Path('/kaggle/working')
_FG_RERUN = bool(_fg_os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or bool(globals().get('IS_RERUN', False))
_FG_T0 = float(globals().get('NB_T0', _fg_time.time()))
FRIGID_DEADLINE = _FG_T0 + (7.5 if _FG_RERUN else 0.6) * 3600.0      # danach startet FRIGID kein neues Molekül
FRIGID_MERGE_UNTIL = _FG_T0 + (7.8 if _FG_RERUN else 0.75) * 3600.0  # so lange wartet die Merge-Zelle höchstens
FRIGID_OUT = _FG_WORK / 'frigid_e1.jsonl'
FRIGID_STATUS = _FG_WORK / 'frigid_status.json'
FRIGID_PROC = None

_FG_PATCH_OLD = "        whl = uniq\n"
_FG_PATCH_NEW = """        whl = uniq
        try:  # Linolium: Wheels für eine andere Python-Version aussortieren (cp312 unter 3.13)
            try:
                from packaging.tags import sys_tags as _st
                from packaging.utils import parse_wheel_filename as _pw
            except ImportError:
                from pip._vendor.packaging.tags import sys_tags as _st
                from pip._vendor.packaging.utils import parse_wheel_filename as _pw
            _ok = set(_st())
            _drop = [Path(x).name for x in whl if not any(t in _ok for t in _pw(Path(x).name)[3])]
            whl = [x for x in whl if Path(x).name not in set(_drop)]
            log(f'wheel filter: kept {len(whl)}, dropped {len(_drop)}: {_drop}')
        except Exception as _e:
            log(f'wheel filter skipped: {_e!r}')
"""

_FG_PATCH2_OLD = "        log(f'site built: {len(whl)} wheels + sdists -> {site}')\n"
_FG_PATCH2_NEW = """        # Linolium: Kaggle entpackt .tar.gz beim Hochladen; omegaconf und antlr4 liegen deshalb nur als Quellordner vor
        # und werden von der *.tar.gz-Schleife nie installiert. Beide sind reines Python: Paketordner direkt kopieren.
        for _pkg, _ver, _rel in (('omegaconf', '2.3.1', 'omegaconf-2.3.1/omegaconf-2.3.1/omegaconf'),
                                 ('antlr4_python3_runtime', '4.9.3', 'antlr4-python3-runtime-4.9.3/antlr4-python3-runtime-4.9.3/src/antlr4')):
            _srcd = assets / 'wheels_extra' / _rel
            _dst = site / _srcd.name
            if _srcd.is_dir() and not _dst.exists():
                shutil.copytree(_srcd, _dst)
                _di = site / f'{_pkg}-{_ver}.dist-info'
                _di.mkdir(exist_ok=True)
                (_di / 'METADATA').write_text(f'Metadata-Version: 2.1\\nName: {_pkg.replace(\"_\", \"-\")}\\nVersion: {_ver}\\n')
                (_di / 'RECORD').write_text('')
                log(f'copied source package {_srcd.name} {_ver}')
            elif not _srcd.is_dir():
                log(f'source package missing: {_rel}')
        log(f'site built: {len(whl)} wheels + sdists -> {site}')
"""

_FG_PATCH3_OLD = "        log('safe-gpt cache restored')\n"
_FG_PATCH3_NEW = """        log('safe-gpt cache restored')
        # Linolium: Abhängigkeitsbericht; Ersatzmodule nur für Pakete, die in Site und Image fehlen
        import base64 as _b64, importlib, importlib.util
        importlib.invalidate_caches()
        _shims = {'loguru': _b64.b64decode('__LOGURU_B64__').decode(), 'ot': _b64.b64decode('__OT_B64__').decode()}
        _rep = {}
        for _m in ('loguru', 'ot', 'networkx', 'h5py', 'lightning_utilities', 'torchmetrics', 'pyarrow', 'regex', 'xxhash',
                   'yaml', 'omegaconf', 'antlr4', 'hydra', 'lightning', 'transformers', 'safe', 'datamol', 'bionemo'):
            try:
                _rep[_m] = importlib.util.find_spec(_m) is not None
            except Exception as _e:
                _rep[_m] = ('error ' + repr(_e))[:80]
            if _rep[_m] is False and _m in _shims:
                (site / _m).mkdir(exist_ok=True)
                (site / _m / '__init__.py').write_text(_shims[_m])
                _rep[_m] = 'shim'
        importlib.invalidate_caches()
        log(f'dependency report: {_rep}')
"""

_FG_WATCH = r"""import os, sys, json, time, subprocess
from pathlib import Path
os.nice(10)
WORK = Path('/kaggle/working')
deadline, rerun = float(sys.argv[1]), sys.argv[2] == '1'
status = WORK / 'frigid_status.json'
def st(**kw):
    status.write_text(json.dumps(dict(kw, t=round(time.time(), 1))))
try:
    runner = next(Path('/kaggle/input').rglob('frigid_runner_self.py'))
    test = next(p for p in Path('/kaggle/input').rglob('test.parquet') if 'mass-spectra' in str(p).lower())
    code = runner.read_text()
    for a, b in json.loads(sys.argv[3]):
        assert code.count(a) == 1, f'runner patch anchor not found exactly once: {a[:60]!r}'
        code = code.replace(a, b)
    patched = WORK / 'frigid_runner_patched.py'
    patched.write_text(code)
    spectra, formulas = str(test), WORK / 'frigid_formulas.tsv'
    # RDKit gehört nicht zum Image: die Pipeline installiert es erst nach dem Start dieses Prozesses
    import importlib, importlib.util, glob
    t_rd = time.time()
    while importlib.util.find_spec('rdkit') is None and time.time() - t_rd < 1200:
        time.sleep(20)
        importlib.invalidate_caches()
    if importlib.util.find_spec('rdkit') is None:
        tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
        whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True)
               if tag in w and os.path.basename(w).startswith('rdkit-2026.3.3-')]
        assert whl, 'RDKit wheel missing'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], check=True)
        importlib.invalidate_caches()
    st(state='rdkit_ready', waited_s=round(time.time() - t_rd))
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    te = pd.read_parquet(test, columns=['molecule_id', 'precursor_mz', 'adduct', 'ionization_mode'])
    if rerun:
        sub_p, rec_p = WORK / 'submission.csv', WORK / 'execution_receipt.json'
        while time.time() < deadline - 1200 and not (sub_p.exists() and rec_p.exists()):
            time.sleep(30)
        if not (sub_p.exists() and rec_p.exists()):
            st(state='no_export_before_deadline'); sys.exit(0)
        time.sleep(20)
        sub = pd.read_csv(sub_p, dtype=str, keep_default_na=False)
        AD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823,
              '[M+K]+': 38.963158, '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402,
              '[M-H2O+H]+': 1.007276 - 18.010565, '[M-2H2O+H]+': 1.007276 - 2 * 18.010565, '[M-H2O-H]-': -1.007276 - 18.010565}
        nm = {}
        for mid, g in te.groupby('molecule_id', sort=False):
            vals = [float(p) - AD[a] for p, a in zip(g.precursor_mz, g.adduct) if a in AD]
            if vals:
                nm[str(mid)] = sorted(vals)[len(vals) // 2]
        rows = []
        for mid, s in zip(sub.molecule_id.astype(str), sub.smiles):
            best = None
            for smi in str(s).split(';')[:5]:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                d = abs(Descriptors.ExactMolWt(m) - nm[mid]) if mid in nm else 0.0
                if best is None or d < best[0]:
                    best = (d, CalcMolFormula(m))
            if best is not None:
                rows.append((mid, best[1]))
        limit = ['--limit', '400']
    else:  # Commit-Lauf: Funktionstest auf zwei Molekülen mit positivem Spektrum
        pos = te[te.adduct.astype(str) == '[M+H]+'].molecule_id.astype(str).unique()[:2]
        rows = [(m, 'C10H12O3') for m in pos]
        limit = ['--limit', '2', '--max-attempts', '16']
    with open(formulas, 'w') as fh:
        fh.write('molecule_id\tformula\n')
        for k, v in rows:
            fh.write(f'{k}\t{v}\n')
    st(state='running', molecules=len(rows))
    env = dict(os.environ, PYTHONPATH=str(WORK / 'frigid_site'), FRIGID_SAFES=str(WORK / 'frigid_safes'),
               HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='1')
    cmd = [sys.executable, str(patched), '--auto-discover', '--setup', '--spectra', spectra, '--formulas', str(formulas),
           '--out', str(WORK / 'frigid_e1.jsonl'), '--deadline', str(deadline), '--device', 'cuda'] + limit
    with open(WORK / 'frigid_runner.log', 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    n = sum(1 for _ in open(WORK / 'frigid_e1.jsonl')) if (WORK / 'frigid_e1.jsonl').exists() else 0
    tail = (WORK / 'frigid_runner.log').read_text(errors='replace')[-2500:]
    st(state='done', returncode=r.returncode, records=n, runner_log_tail=tail)
except Exception as e:
    st(state='error', error=repr(e)[:400])
"""

try:
    (_FG_WORK / 'frigid_watch.py').write_text(_FG_WATCH)
    FRIGID_PROC = _fg_sp.Popen([_fg_sys.executable, str(_FG_WORK / 'frigid_watch.py'), str(FRIGID_DEADLINE),
                                '1' if _FG_RERUN else '0',
                                _fg_json.dumps([[_FG_PATCH_OLD, _FG_PATCH_NEW], [_FG_PATCH2_OLD, _FG_PATCH2_NEW],
                                            [_FG_PATCH3_OLD, _FG_PATCH3_NEW]])],
                               stdout=open(_FG_WORK / 'frigid_watch.log', 'w'), stderr=_fg_sp.STDOUT)
    print('[frigid] watcher started, pid', FRIGID_PROC.pid, '| rerun', _FG_RERUN,
          '| deadline at %.2f h' % ((FRIGID_DEADLINE - _FG_T0) / 3600), flush=True)
except Exception as _fg_exc:
    print('[frigid] watcher not started:', repr(_fg_exc)[:300], flush=True)
'''

MERGE_MD = """## Linolium · Baustein B: FRIGID-Kandidaten (Zusammenführung, Rang {pos})

Für Listen ohne starken Bibliothekstreffer (`library_max` < 0.90) kommt der erste FRIGID-Vorschlag mit
Fingerprint-Ähnlichkeit ≥ 0.70 und höchstens 30 schweren Atomen auf **Rang {pos}**, sofern er nicht schon weiter oben steht.
Rang 1 wird nie geändert; bei jedem Fehler bleibt die Abgabe unverändert."""

MERGE_CELL = r'''# ── Linolium · Baustein B: FRIGID-Kandidaten zusammenführen (Rang __POS__) ──
import json as _fm_json, os as _fm_os, time as _fm_time, traceback as _fm_tb
from pathlib import Path as _FM_Path

_FM_POS = int('__POS__')     # Einfügeposition (2 = direkt hinter Rang 1)
_FM_GATE, _FM_MIN_SIM, _FM_MAX_HEAVY = 0.90, 0.70, 30


def _fm_merge(work):
    import pandas as pd
    from rdkit import Chem
    sp, p = work / 'submission.csv', work / 'frigid_e1.jsonl'
    if not p.exists() or not sp.exists():
        return dict(status='nothing to merge')
    recs = {}
    for line in open(p):
        try:
            r = _fm_json.loads(line)
        except Exception:
            continue
        if r.get('preds'):
            recs[str(r.get('molecule_id'))] = r
    sub = pd.read_csv(sp, dtype=str, keep_default_na=False)
    lib = {}
    ev = work / 'frozen_candidate_evidence.parquet'
    if ev.exists():
        d = pd.read_parquet(ev, columns=['molecule_id', 'library_max'])
        lib = {str(m): float(v) for m, v in d.groupby('molecule_id', sort=False).library_max.first().items()}
    cache = {}
    def key14(s):
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            k = Chem.MolToInchiKey(m) if m is not None else ''
            cache[s] = k[:14] if k else None
        return cache[s]
    n = dict(records=len(recs), library_gate=0, no_good_pred=0, already_top=0, inserted=0, moved=0)
    out = []
    for mid, s in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = [x for x in str(s).split(';') if x]
        r = recs.get(mid)
        if r and len(lst) >= _FM_POS - 1:
            if lib.get(mid, 0.0) >= _FM_GATE:
                n['library_gate'] += 1
            else:
                best = None
                for item in r['preds'][:10]:
                    smi, sim = item[0], float(item[2])
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None and mol.GetNumHeavyAtoms() <= _FM_MAX_HEAVY and sim >= _FM_MIN_SIM:
                        best = smi
                        break
                k = key14(best) if best else None
                if k is None:
                    n['no_good_pred'] += 1
                else:
                    keys = [key14(x) for x in lst]
                    if k in keys[:_FM_POS]:
                        n['already_top'] += 1
                    else:
                        if k in keys:
                            lst.pop(keys.index(k)); n['moved'] += 1
                        else:
                            lst = lst[:24] if len(lst) >= 25 else lst; n['inserted'] += 1
                        lst.insert(_FM_POS - 1, best)
        out.append(';'.join(lst[:25]))
    new = sub.copy(); new['smiles'] = out
    assert len(new) == len(sub) and (new.molecule_id == sub.molecule_id).all()
    assert all(a.split(';')[0] == b.split(';')[0] for a, b in zip(new.smiles, sub.smiles)), 'rank 1 changed'
    assert new.smiles.str.split(';').map(len).max() <= 25
    sub.to_csv(work / 'submission_before_frigid.csv', index=False)
    new.to_csv(str(sp) + '.fm_tmp', index=False); _fm_os.replace(str(sp) + '.fm_tmp', sp)
    return dict(status='applied', **n)


_fm_work = _FM_Path('/kaggle/working')
_fm_rerun = bool(_fm_os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or bool(globals().get('IS_RERUN', False))
_fm_until = float(globals().get('FRIGID_MERGE_UNTIL', _fm_time.time()))
_fm_proc = globals().get('FRIGID_PROC')
while _fm_proc is not None and _fm_proc.poll() is None and _fm_time.time() < _fm_until:
    _fm_time.sleep(15)
_fm_status = {}
try:
    _fm_status = _fm_json.loads((_fm_work / 'frigid_status.json').read_text())
except Exception:
    pass
print('[frigid] watcher status', _fm_json.dumps(_fm_status), flush=True)
if _fm_rerun:
    _fm_bytes = (_fm_work / 'submission.csv').read_bytes() if (_fm_work / 'submission.csv').exists() else None
    try:
        print('[frigid] merge', _fm_json.dumps(_fm_merge(_fm_work)), flush=True)
    except Exception:
        _fm_tb.print_exc()
        if _fm_bytes is not None:
            (_fm_work / 'submission.csv').write_bytes(_fm_bytes)
        print('[frigid] merge failed; submission restored', flush=True)
else:
    # Commit-Lauf: der Funktionstest muss Vorschläge erzeugt haben, sonst absichtlich fehlschlagen
    _fm_n = 0
    if (_fm_work / 'frigid_e1.jsonl').exists():
        _fm_n = sum(1 for l in open(_fm_work / 'frigid_e1.jsonl') if '"preds": [[' in l)
    print('[frigid] smoke test records with predictions:', _fm_n, flush=True)
    if _fm_n < 1:
        raise RuntimeError(f'FRIGID smoke test failed: {_fm_status}')
'''


def shorten_markdown(nb):
    first = True
    for c in nb["cells"]:
        if c["cell_type"] != "markdown":
            continue
        s = "".join(c["source"])
        if first:
            first = False
            continue
        if len(s) > 400 and "Credits" not in s.split("\n")[0] and not s.startswith("## Linolium"):
            head = s.split("\n")[0]
            c["source"] = f"{head}\n\n*(Text gekürzt wegen Kaggles 1-MB-Grenze; vollständig im [Original]({ORIGINAL}).)*\n"


def build(folder, pos, slug, title):
    nb = json.loads((SRC / "vt_0509.ipynb").read_text())
    code_before = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
           and "BUNDLE = materialise_sources()" in "".join(c["source"])]
    assert len(idx) == 1, f"Pipeline-Zelle {len(idx)}x gefunden"
    assert "VIRTUAL NATURAL-PRODUCT TIER" in "".join(nb["cells"][-1]["source"]), "letzte Zelle ist nicht der VT-Tier"
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    cc = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    watch = (WATCH_CELL.replace("__LOGURU_B64__", base64.b64encode(LOGURU_SHIM.encode()).decode())
             .replace("__OT_B64__", base64.b64encode(OT_SHIM.encode()).decode()))
    nb["cells"][idx[0]:idx[0]] = [md(WATCH_MD), cc(watch)]
    nb["cells"] += [md(MERGE_MD.format(pos=pos)), cc(MERGE_CELL.replace("__POS__", str(pos)))]
    shorten_markdown(nb)
    code_after = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert [s for s in code_after if s not in (watch, MERGE_CELL.replace("__POS__", str(pos)))] == code_before
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"linuszarse/{slug}", title=title, code_file=f"{folder}.ipynb", is_private=True)
    for d in FRIGID_DATASETS:
        if d not in meta["dataset_sources"]:
            meta["dataset_sources"].append(d)
    out = ROOT / folder
    out.mkdir(exist_ok=True)
    text = json.dumps(nb, ensure_ascii=False)
    (out / f"{folder}.ipynb").write_text(text)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", out, "| Bytes", len(text.encode()))


if __name__ == "__main__":
    for folder, (pos, slug, title) in VARIANTS.items():
        build(folder, pos, slug, title)
