"""Erzeugt kaggle/public_701/v9_postapex/ – V6 (0.707) + Apex-Nachbearbeitung auf den Logits nach der GLM-Stufe.

Basis: v6_hgb/v6_hgb.ipynb (alexchilton 0.701 + unser HGB-Klassifikator, Public LB 0.707), unverändert von Kaggle.
Wie V8 auf der 0.702-Basis (Experiment 09): Die Apex-Zelle rechnet bisher mit den Stack-Logits vor der GLM-Stufe
und ersetzt jede Nicht-GLM-Vorhersage. Was die GLM-Stufe von GLM weg verschiebt, geht damit verloren.

Zwei Änderungen, sonst nichts:
1. v1_stage.py speichert zusätzlich post-glm.npz (ids, labels, logits) in V1_OUT: zuerst die Stack-Logits
   (gilt, falls die Stufe aus Zeitgründen aussteigt), nach dem Mischen die Logits zl + w * ll. Die Vorhersagen
   der Stufe bleiben bitgleich (argmax derselben Summe).
2. Die Apex-Zelle liest post-glm.npz statt FINAL_LOGITS, falls vorhanden (sonst wie bisher FINAL_LOGITS).
   GLM-Labels bleiben wie im Original erhalten.

Die GLM-Stufe von V6 deckt so viele Texte ab, wie bis 8,8 h passen (höchstens 30 %), statt fest 15 % wie im
0.702-Notebook. In der Simulation (experiments/09_postprocess) bringt die Korrektur bei 20 % Abdeckung +0,55 Pp.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v9_postapex.py
    .venv/bin/kaggle kernels push -p kaggle/public_701/v9_postapex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "public_701"
BASE = SRC / "v6_hgb" / "v6_hgb.ipynb"
OUT = SRC / "v9_postapex"
SAVE = "np.savez(Path(a.outdir) / 'post-glm.npz', ids=np.asarray(ids), labels=np.asarray(labels), logits={})"

# (alt, neu) in v1_stage.py; jedes alt muss genau einmal vorkommen
STAGE_EDITS = [
    ("    zl = log_softmax(z, 1)\n",
     "    zl = log_softmax(z, 1)\n"
     "    Path(a.outdir).mkdir(parents=True, exist_ok=True)   # ours (V9)\n"
     "    " + SAVE.format("zl") + "   # ours (V9): fallback = stack logits\n"),
    ("    after[sel] = (zl[sel] + w * ll).argmax(1)\n",
     "    post_ll = zl.copy()   # ours (V9): keep the blended logits for the Apex post-processing\n"
     "    post_ll[sel] += w * ll\n"
     "    after[sel] = post_ll[sel].argmax(1)\n"),
    ("    sub = pd.DataFrame({'id': ids, 'model': np.asarray(labels)[after]})\n",
     "    " + SAVE.format("post_ll") + "   # ours (V9)\n"
     "    sub = pd.DataFrame({'id': ids, 'model': np.asarray(labels)[after]})\n"),
]

PATCH = '''# ours (V9): v1_stage.py also saves the logits after the GLM stage (post-glm.npz in V1_OUT): the stack logits
# first (kept if the stage falls back), then zl + w * ll after the blend. The stage's predictions are unchanged.
# The Apex cell at the end reads them instead of FINAL_LOGITS, so it no longer overwrites the GLM stage.
_stage = V1_DIR / 'v1_stage.py'
_src = _stage.read_text()
for _old, _new in __EDITS__:
    assert _src.count(_old) == 1, f'V9: v1_stage.py is not the expected one: {_old!r}'
    _src = _src.replace(_old, _new)
compile(_src, str(_stage), 'exec')
_stage.write_text(_src)
print('V9: v1_stage.py saves post-glm.npz', flush=True)
'''

APEX_OLD = ("        if os.path.exists(FINAL_LOGITS):\n"
            "            z_raw = np.load(FINAL_LOGITS)\n")
APEX_NEW = ("        # ours (V9): the logits after the GLM stage, so Apex keeps its corrections (fallback: FINAL_LOGITS)\n"
            "        _base_file = str(V1_OUT / 'post-glm.npz') if (V1_OUT / 'post-glm.npz').exists() else FINAL_LOGITS\n"
            "        print('Apex base logits:', _base_file, flush=True)\n"
            "        if os.path.exists(_base_file):\n"
            "            z_raw = np.load(_base_file)\n"
            "            if 'ids' in z_raw:\n"
            "                assert z_raw['ids'].astype(str).tolist() == query_df.id.astype(str).tolist(), 'Apex: row order'\n")


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def build():
    nb = json.loads(BASE.read_text())
    meta = json.loads((SRC / "v6_hgb" / "kernel-metadata.json").read_text())
    cells = [src(c) for c in nb["cells"]]

    def one(pred, what):
        hits = [i for i, s in enumerate(cells) if pred(s)]
        assert len(hits) == 1, f"{what}: {len(hits)} Treffer – V6-Notebook hat sich geändert"
        return hits[0]

    i_out = one(lambda s: "V1_OUT = Path('/kaggle/working/v1'); V1_OUT.mkdir(" in s, "V1_OUT")
    i_hgb = one(lambda s: s.startswith("# ours (V6): GLM likelihood classifier"), "HGB-Patch")
    i_run = one(lambda s: "'v1_stage.py'), '--test-csv'" in s and "subprocess.call(_cmd" in s, "Stufe starten")
    i_apex = one(lambda s: APEX_OLD in s, "Apex-Zelle")
    assert i_out < i_hgb < i_run < i_apex, (i_out, i_hgb, i_run, i_apex)
    assert cells[i_apex].count(APEX_OLD) == 1 and "except Exception" in cells[i_apex]

    # Ersetzungen gegen den echten v1_stage.py aus dem Payload prüfen (sonst fiele es erst auf Kaggle auf)
    import base64, io, re, tarfile
    s28 = cells[i_out]
    m = re.search(r"V1_PAYLOAD\s*=\s*\(?\s*((?:['\"][A-Za-z0-9+/=]*['\"]\s*)+)\)?", s28)
    raw = base64.b64decode("".join(re.findall(r"['\"]([A-Za-z0-9+/=]*)['\"]", m.group(1))))
    stage = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz").extractfile("v1_stage.py").read().decode()
    for old, new in STAGE_EDITS:
        assert stage.count(old) == 1, old
        stage = stage.replace(old, new)
    compile(stage, "v1_stage.py", "exec")

    code = PATCH.replace("__EDITS__", repr(STAGE_EDITS))
    compile(code, "patch", "exec")
    nb["cells"].insert(i_hgb + 1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                   "source": code})
    cell = nb["cells"][i_apex + 1]   # +1: Patch-Zelle davor eingefügt
    cell["source"] = src(cell).replace(APEX_OLD, APEX_NEW)

    assert src(nb["cells"][0]).startswith("# Fork V6")
    nb["cells"][0]["source"] = (
        "# Fork V9: 0.701 + GLM-Likelihood-Klassifikator + Apex nach der GLM-Stufe\n\n"
        "Wie V6 (0.701 von alexchilton + unser GLM-Klassifikator: Gradient Boosting auf train.csv + 5.670 "
        "Open-Pool-Texten, bewertet unter GLM-5.3 und GLM-5.3-Flash). Zusätzlich speichert die GLM-Stufe ihre "
        "Logits, und die Apex-Nachbearbeitung am Ende rechnet auf diesen statt auf den Logits vor der GLM-Stufe. "
        "Sonst überschreibt Apex die Korrekturen der GLM-Stufe. Alle Credits unten erhalten.")
    OUT.mkdir(exist_ok=True)
    (OUT / "v9_postapex.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    meta.update(id="linuszarse/ntl-701-fork-v9-postapex", title="NTL 701 Fork v9 postapex",
                code_file="v9_postapex.ipynb", is_private=True)
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"geschrieben: {OUT}/v9_postapex.ipynb, Patch-Zelle nach Zelle {i_hgb}, Apex-Zelle {i_apex + 1}")


if __name__ == "__main__":
    build()
