"""Erzeugt kaggle/public_701/v10_w15/ – V9 + stärkeres Mischgewicht der GLM-Stufe + Abdeckung höchstens 25 %.

Basis: v9_postapex/v9_postapex.ipynb (V6 0.707 + Apex auf den Logits nach der GLM-Stufe). Zwei Änderungen:
1. Mischgewicht des GLM-Klassifikators W = 1.5 statt 1.0 (Patch-Zelle von V6, Wert landet in ll_model.json).
2. Die GLM-Stufe bewertet höchstens 25 % des Tests (--max-cover 0.25 statt Standard 0.30); weniger, wenn die Zeit
   nicht reicht (Deadline unverändert 8,8 h).

Simulation (V9-Kette auf train.csv, Stack-Ersatz, Experiment 09, 17.11.):
             W=1.0    W=1.25   W=1.5    W=2.0
  cover 20 % 0.6434   0.6431   0.6453   0.6431
  cover 25 % 0.6472   0.6475   0.6494   0.6468
  cover 30 % 0.6460   0.6449   0.6472   0.6423
W = 1.5 ist bei jeder Abdeckung vorn (+0,2 Pp.), 25 % vor 30 %. Die Abbruchgrenze der Stufe (25 % geänderte
Vorhersagen aller Texte) bleibt weit entfernt: im Nachbau 11,3 % bei W 1.5 / 25 %.

V11 (--variant v11) ist dieselbe Änderung auf V6 statt V9, also ohne die Apex-Korrektur: Absicherung, falls diese
auf dem echten Stack schadet. Simulation V6-Kette 25 %: W 1.0 0.6390 → W 1.5 0.6405 (V9-Kette: 0.6472 → 0.6494).

Aufruf (aus Name_That_LLM_2026/):
    .venv/bin/python kaggle/public_701/build_v10_w15.py [--variant v11]
    .venv/bin/kaggle kernels push -p kaggle/public_701/v10_w15      # bzw. v11_v6w15
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kaggle" / "public_701"
VARIANTS = {   # name: (Basisordner, Basis-Notebook, Kopf der Basis, Ausgabeordner, Kernel-Slug, Beschreibung)
    "v10": ("v9_postapex", "v9_postapex.ipynb", "# Fork V9", "v10_w15", "ntl-701-fork-v10-w15",
            "# Fork V10: 0.701 + GLM-Likelihood-Klassifikator + Apex nach der GLM-Stufe, W 1.5, Abdeckung ≤ 25 %\n\n"
            "Wie V9 (0.701 von alexchilton + unser GLM-Klassifikator: Gradient Boosting auf train.csv + 5.670 Open-Pool-"
            "Texten unter GLM-5.3 / GLM-5.3-Flash; Apex rechnet auf den Logits nach der GLM-Stufe)."),
    "v11": ("v6_hgb", "v6_hgb.ipynb", "# Fork V6", "v11_v6w15", "ntl-701-fork-v11-v6w15",
            "# Fork V11: 0.701 + GLM-Likelihood-Klassifikator, W 1.5, Abdeckung ≤ 25 %\n\n"
            "Wie V6 (0.701 von alexchilton + unser GLM-Klassifikator: Gradient Boosting auf train.csv + 5.670 Open-Pool-"
            "Texten unter GLM-5.3 / GLM-5.3-Flash), Apex unverändert."),
}
W, COVER = 1.5, 0.25
EDITS = [
    ("_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W=1.0,",
     f"_m.update(kind='hgb', clf_file=str(V1_DIR / 'll_hgb.pkl'), W={W},   # ours (V10): W 1.0 -> {W}"),
    ("    _cmd += ['--pack'] if V1_PACK else []\n",
     f"    _cmd += ['--max-cover', '{COVER}']   # ours (V10): at most {COVER:.0%} of the test (default 0.30)\n"
     "    _cmd += ['--pack'] if V1_PACK else []\n"),
]


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def build(variant):
    base, base_nb, head, out, slug, desc = VARIANTS[variant]
    nb = json.loads((SRC / base / base_nb).read_text())
    meta = json.loads((SRC / base / "kernel-metadata.json").read_text())
    for old, new in EDITS:
        hits = [c for c in nb["cells"] if old in src(c)]
        assert len(hits) == 1 and src(hits[0]).count(old) == 1, f"nicht eindeutig: {old!r}"
        hits[0]["source"] = src(hits[0]).replace(old, new)
        compile(src(hits[0]), "cell", "exec")
    assert src(nb["cells"][0]).startswith(head)
    nb["cells"][0]["source"] = (desc + f" Zusätzlich: Mischgewicht des Klassifikators W = {W} statt 1.0, GLM-Stufe "
                                f"bewertet höchstens {COVER:.0%} des Tests. Alle Credits unten erhalten.")
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
    print(f"geschrieben: {SRC / out}/{out}.ipynb ({size / 2**10:.0f} KB), W={W}, max-cover={COVER}, Titel {meta['title']!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="v10")
    build(ap.parse_args().variant)
