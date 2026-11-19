"""Erzeugt V6-Varianten, die nur Stellschrauben ändern (Basis: v6_hgb/v6_hgb.ipynb, die abgegebene V6, 0.707).

Seit der V6-Wiederholung (18.11.: 0.707 wie das Original) wissen wir: Die Pipeline reproduziert auf etwa ±0,001.
Kleine LB-Unterschiede sind also echt (bezogen auf den Public-Teil). Deshalb lohnen gezielte Proben einzelner Werte,
die nie am LB geprüft wurden:
- --sinkhorn S        Stärke des Sinkhorn-Ausgleichs im Stack (V6: 0.30)
- --target-mix R      Anteil des Train-Priors im Sinkhorn-Ziel (V6: 0.35)
- --apex-strength X   Stärke der Geschwister-Verfeinerung in Apex (V6: 0.10)

Jede Option ist eine eindeutige Textersetzung (Assertion: genau ein Treffer); der Rest bleibt bitgleich.

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v6_knobs.py --name v15_sk06 --sinkhorn 0.6
    .venv/bin/kaggle kernels push -p kaggle/public_701/v15_sk06
"""
import argparse
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
BASE = SRC / "v6_hgb"


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def build(name, sinkhorn=None, target_mix=None, apex_strength=None):
    nb = json.loads((BASE / "v6_hgb.ipynb").read_text())
    meta = json.loads((BASE / "kernel-metadata.json").read_text())
    edits, desc = [], []
    if sinkhorn is not None:
        edits.append(("'--sinkhorn', '0.30'", f"'--sinkhorn', '{sinkhorn}'"))
        desc.append(f"Sinkhorn strength 0.30 -> {sinkhorn}")
    if target_mix is not None:
        edits.append(("'--sinkhorn_target_mix', '0.35'", f"'--sinkhorn_target_mix', '{target_mix}'"))
        desc.append(f"Sinkhorn target mix 0.35 -> {target_mix}")
    if apex_strength is not None:
        edits.append(("def refine(log_probability, evidence, families, strength=0.10):",
                      f"def refine(log_probability, evidence, families, strength={apex_strength}):   # ours: 0.10 -> {apex_strength}"))
        desc.append(f"Apex sibling strength 0.10 -> {apex_strength}")
    assert edits, "keine Änderung angegeben"
    cells = [src(c) for c in nb["cells"]]
    for old, new in edits:
        hits = [i for i, s in enumerate(cells) if old in s]
        assert len(hits) == 1 and cells[hits[0]].count(old) == 1, (old, hits)
        cells[hits[0]] = cells[hits[0]].replace(old, new)
        body = cells[hits[0]].split("\n", 1)[1] if cells[hits[0]].startswith("%%writefile") else cells[hits[0]]
        compile(body, "cell", "exec")   # die geänderte Zelle muss gültiges Python bleiben
    for c, s in zip(nb["cells"], cells):
        c["source"] = s
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    assert cells[0].startswith("# Fork V6")
    nb["cells"][0]["source"] = (
        f"# Fork {name.split('_')[0].upper()}: V6 (0.707) with one knob changed\n\n"
        f"Same as V6 (public 0.701 by alexchilton + our HGB GLM likelihood classifier). Only change: "
        f"{'; '.join(desc)}. All credits below kept.")
    out = SRC / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta.pop("id_no", None)
    slug = f"ntl-701-fork-{name.replace('_', '-')}"
    meta.update(id=f"linuszarse/{slug}", title=f"NTL 701 Fork {name.replace('_', ' ')}", code_file=f"{name}.ipynb",
                is_private=True)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"geschrieben: {out}/{name}.ipynb | {slug} | {'; '.join(desc)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sinkhorn", type=float)
    ap.add_argument("--target-mix", type=float)
    ap.add_argument("--apex-strength", type=float)
    a = ap.parse_args()
    build(a.name, a.sinkhorn, a.target_mix, a.apex_strength)
