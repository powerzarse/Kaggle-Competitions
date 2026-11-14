"""Erzeugt kaggle/network_0488_gnps/ aus kaggle/network_0488/: das öffentliche 0.488-Notebook plus eine dritte
Referenzbibliothek im Rang-1-Abgleich (Zelle "APEX OVERLAY").

Änderung: Source C = GNPS-Community-Bibliotheken (CC0, aryanawasthiteykik/casmi26-gnps-community-cc0), nur Spektren
von Strukturen, die NICHT in train.parquet stehen. Für Strukturen aus dem Training hat die Pipeline schon eigene
Bibliothekstreffer; neu sind nur Kandidaten, die sie sonst gar nicht kennen könnte. Gleiche Regel wie Source A/B:
Präkursor <= 10 ppm + 0.002 Da, gleicher Ionenmodus, Kosinus >= 0.80, >= 4 passende Peaks, und keine Änderung, wenn
die Pipeline selbst schon einen Bibliothekstreffer >= 0.90 hat. Jeder Fehler lässt die Abgabe unverändert.

Aufruf (aus Enveda_CASMI_2026/):
    python kaggle/build_variant_gnps.py
    ../tools/kg kernels push -p kaggle/network_0488_gnps
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "network_0488"
DST = ROOT / "network_0488_gnps"
GNPS_DATASET = "aryanawasthiteykik/casmi26-gnps-community-cc0"

ANCHOR = "    if not refs:\n        print('[apex-extlib] No external reference spectra available; keeping submission untouched.')"

SOURCE_C = '''    # Source C (Linolium): GNPS community libraries, CC0, only structures absent from train.parquet
    g_hits = sorted(glob.glob(inp + '/**/gnps_cc0_trainschema.parquet', recursive=True), key=len)
    tr_hits = sorted(glob.glob(inp + '/**/train.parquet', recursive=True),
                     key=lambda p: ('mass-spectra' not in p.lower(), len(p)))
    if g_hits and tr_hits:
        try:
            import pyarrow.parquet as _pq
            _train_keys = set(_pq.read_table(tr_hits[0], columns=['inchikey14']).column(0).to_pylist())
            g_df = pd.read_parquet(g_hits[0], columns=['inchikey14', 'normalized_smiles', 'ionization_mode',
                                                       'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
            g_df = g_df[g_df.inchikey14.notna() & ~g_df.inchikey14.isin(_train_keys)]
            n_before = len(refs)
            for row in g_df.itertuples(index=False):
                try:
                    mode = 1 if 'pos' in str(row.ionization_mode).lower() else -1
                    prec = float(row.precursor_mz)
                    mz_p = np.asarray(row.ms2_mzs, dtype=np.float64)
                    it_p = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
                    mask = (mz_p > 0) & (it_p > 0)
                    mz_p, it_p = mz_p[mask], it_p[mask]
                    if len(mz_p) < 3:
                        continue
                    o = np.argsort(mz_p)
                    refs.append((mode, prec, mz_p[o], it_p[o], str(row.inchikey14), str(row.normalized_smiles),
                                 prec - 1.007276 * mode))
                except Exception:
                    continue
            print(f'[apex-extlib] Source C (GNPS CC0, not in train): {len(refs) - n_before} spectra, '
                  f'{g_df.inchikey14.nunique()} structures.')
        except Exception as e:
            print(f'[apex-extlib] Source C loading error: {e}')

'''


def main():
    nb = json.loads((SRC / "network_0488.ipynb").read_text())
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and ANCHOR in "".join(c["source"])]
    assert len(hits) == 1, f"Anker {len(hits)}x gefunden, erwartet 1x"
    cell = hits[0]
    src = "".join(cell["source"])
    assert src.count(ANCHOR) == 1
    cell["source"] = src.replace(ANCHOR, SOURCE_C + ANCHOR)

    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="linuszarse/casmi26-network-0488-gnps", title="CASMI26 Network 0488 GNPS",
                code_file="network_0488_gnps.ipynb", is_private=True)
    if GNPS_DATASET not in meta["dataset_sources"]:
        meta["dataset_sources"].append(GNPS_DATASET)

    DST.mkdir(exist_ok=True)
    (DST / "network_0488_gnps.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("geschrieben:", DST)


if __name__ == "__main__":
    main()
