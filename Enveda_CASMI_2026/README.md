# Enveda CASMI 2026 – Molecule ID From Mass Spectra (Kaggle)

Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra

## Aufgabe
Aus MS/MS-Spektren die 2D-Struktur eines Moleküls vorhersagen. Je Molekül bis zu 25 SMILES-Kandidaten, bester zuerst.

- **Metrik:** MRR@25. Ein Treffer zählt, wenn der InChIKey14 (Atom-Konnektivität) übereinstimmt.
- **Test:** ~400 Moleküle mit ~1.500 Spektren (Bruker timsTOF), versteckt. Ein Molekül auf Rang 1 ist also 0,0025 Punkte wert.
- **Format:** Code-Competition, Notebook ≤ 9 h, kein Internet, öffentliche externe Daten und Modelle erlaubt.
- **Termine:** Beitritt und Team-Merge bis 07.12.2026, letzte Abgabe 14.12.2026 (23:59 UTC).

## Stand (12.11.2026)
- 5.183 Teams. Bronze-Grenze 0.487 (Rang 518), Gold 0.497 (Rang 20).
- 607 Teams stehen exakt auf 0.487, dem Score des besten öffentlichen Notebooks.
- Laut dessen Autoren schwankt derselbe Code von Lauf zu Lauf um etwa 0.002, und wortgleiche Kopien landen bis zu 0.01 auseinander. Das Public Leaderboard ist also stark verrauscht.

## Struktur
```
Enveda_CASMI_2026/
├── kaggle/
│   ├── baseline_public_0487/   öffentliches 0.487-Notebook (Stand 12.11.)
│   ├── network_0488/           öffentliches 0.488-Notebook, unverändert (Abgabe 1)
│   ├── network_0488_gnps/      0.488 + GNPS-CC0 als dritte Overlay-Bibliothek (Abgabe 2)
│   └── build_variant_gnps.py   erzeugt network_0488_gnps/ aus network_0488/
└── data/                       nicht im Git
```

## Analyse (14.11.2026)
- **Medaillengrenzen Public LB:** Bronze 0.488, Silber 0.489. Bei Gleichstand gewinnt die frühere Abgabe, und 0.487 bzw. 0.488 sind schon voll belegt.
- **Rauschen:** Das Public LB bewertet nur ~125–130 Moleküle. Ein Molekül von Rang 2 auf Rang 1 ist dort ~0.004 wert, also mehr als der Abstand zwischen Bronze und Silber.
- **Bestes öffentliches Notebook (heliosli, 0.489):** nicht reproduzierbar. Seine beiden Zusatzstufen (FRIGID-de-novo-Kandidaten auf Rang 2, rr3-Isomer-Reranker) laden Modelle aus nicht öffentlichen Datensätzen und werden ohne sie übersprungen.
- **Weitere Bibliotheken:** Gegenüber train.parquet und den bisherigen Overlay-Quellen (extlib, MassBank harmonized) bringt GNPS-CC0 nur 1.141 Strukturen, die nicht im Training stehen (37 auch in keiner Overlay-Quelle). Die neue MassBank-Version bringt keine. Erwartete Wirkung von Abgabe 2 daher klein.

## Baseline
`kaggle/baseline_public_0487/` ist eine unveränderte Kopie (Ausgaben entfernt) von
[CASMI26 LB 0.487 | Peaks to Structures + Scorer](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-487-peaks-to-structures-scorer)
von alexchilton, das seinerseits auf den im Notebook genannten Autoren aufbaut.

## Kaggle-Zugriff in der Cloud-Session
`tools/kg` (im Repo-Root) ist ein Wrapper um die Kaggle-CLI, der über `www.kaggle.com` spricht.
Der Token liegt nur im Netzwerk-Proxy der Umgebung „Kaggle-Competitions“.

```bash
pip install kaggle
../tools/kg competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data
../tools/kg kernels push -p kaggle/baseline_public_0487
```
