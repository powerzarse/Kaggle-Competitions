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
│   ├── heliosli_0495/          0.495-Demo von heliosli (nicht abgebbar, siehe unten)
│   ├── network_lam035_0492/    öffentliches 0.492-Notebook (mathleteutd, LAM 0.35), unverändert
│   ├── anchored_rank2_pin/     0.492 + Rang-2-Absicherung (flexonafft v7), unverändert
│   ├── network_fr_p5_0495/     0.495 von taeyangg4 (Netzwerk D-L1 + Fragment-Reranker + P5-Tail), unverändert
│   ├── build_variant_r1judge.py  erzeugt r1judge_gap150/ und r1judge_gap075/ (Baustein A)
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

## Abgaben
| Datum | Notebook | Public LB |
|---|---|---|
| 14.11. | `network_0488` (unverändert) | 0.488 |
| 14.11. | `network_0488_gnps` | 0.488 |
| 17.11. | `heliosli_0495` | abgelehnt (nicht versionierte Datensätze) |
| 18.11. | `anchored_rank2_pin` | 0.492 |
| 18.11. | `network_lam035_0492` | 0.492 |
| 18.11. | `network_fr_p5_0495` | **0.495** (Rang 132 von 5.528, Silberbereich) |
| 19.11. | `r1judge_gap150` (Baustein A, streng) | 0.495 |
| 19.11. | `r1judge_gap075` (Baustein A, locker) | 0.495 |

## Stand 17.11.2026
- 5.448 Teams. Bronze-Grenze 0.492 (Rang 544), Silber 0.492 (Rang 272), Gold 0.504 (Rang 20).
- 513 Teams stehen genau auf 0.492. Wegen des Zeit-Tiebreaks braucht eine neue Abgabe mindestens 0.493 für Bronze.
- Mit 0.488 von Rang 412 (15.11.) auf Rang 767 gefallen.
- heliosli's 0.495-Notebook ist nicht nutzbar: Kaggle lehnt die Abgabe ab, weil seine sieben Datensätze nicht versioniert sind. Wichtiger noch: Laut deren Beschreibung enthalten sie statt der trainierten Gewichte nur zufällig initialisierte Demo-Gewichte und reproduzieren das Ergebnis nicht. Kopien hochzuladen lohnt sich daher nicht.
- Stattdessen zwei vollständig öffentliche Notebooks der 0.492-Linie: Testset-Netzwerk mit `LAM` 0.35 (Messreihe LAM 0/0.15/0.25/0.35/0.50 → 0.487/0.487/0.488/0.492/0.484) und dieselbe Basis plus Rang-2-Absicherung: Wird der Spitzenkandidat der letzten Stufe von einer späteren Stufe verdrängt, landet er auf Rang 2 statt auf Rang 3–5. Rang 1 ändert sich nie.

## Stand 18.11.2026
- 5.516 Teams, eigener Rang 905 (0.488). 644 Teams genau auf 0.492, nur 132 Teams ≥ 0.493.
- Silber-Grenze Rang 275 (0.492): Jede Abgabe ≥ 0.493 liegt damit im Silberbereich, 0.492 reicht für keine Medaille.
- GPU-Kontingent: 6 h pro Woche, aktuell 11,7 h verbraucht (Reset 21.11.). Fast-Commits liefen trotzdem durch.
- Bester reproduzierbarer Kandidat: taeyangg4 0.495. Beide eigenen Datensätze des Autors sind öffentlich und versioniert und enthalten echte Gewichte (Fragment-Reranker, 768k Parameter, Trainingsprotokoll). Jede Stufe einzeln belegt: 0.487 → 0.487 → 0.488 → 0.490 → 0.494 → 0.495.

## Stand 19.11.2026
- **Rang 132 von 5.528 mit 0.495** (taeyangg4-Notebook, reproduziert). Silber reicht bis Rang 276, Bronze bis 552, Gold ab 0.505 (Rang 21).
- Die Rang-2-Absicherung hat auf der 0.492-Basis nichts bewegt (beide 0.492).
- Vorläufig: Medaillen vergibt am 14.12. das private Leaderboard (~270 Moleküle).

## Gold-Plan (ab 19.11.2026)
Ziel: Gold, also ≥ 0.505 public bzw. Top 21 privat. Abstand zu 0.495: +0.010, auf dem Public LB 2–3 Moleküle.

**Was frühere Versuche anderer gezeigt haben** (flexonafft, 18 Versionen mit je einer Änderung): Änderungen, die kalibrierte
Teile in Ruhe lassen, enden unentschieden. Andere Eingaben für kalibrierte Teile (LightGBM-Ranker, PubChem-Kanal) und
zusätzliche Bibliotheken direkt auf Rang 1 verlieren (-0.009 bis -0.015). Gewinne kamen nur von **neuen Signalen**.

**Wo Punkte verloren gehen** und welcher Baustein ansetzt:

| Fehlerart | Baustein | Status |
|---|---|---|
| Richtiger Kandidat in der Liste, aber nicht auf Rang 1 | **A · Rang-1-Richter**: FR-Scores entscheiden eng begrenzt über Rang 1 ↔ 2 | beide Dosierungen 0.495 = Kontrolle, auf dem Public LB kein Effekt |
| Richtiger Kandidat fehlt ganz (neue Strukturen, Klasse 3) | **B · de-novo-Kandidaten**: öffentliche FRIGID-Checkpoints, bester Vorschlag auf Rang 2 nur bei hoher Fingerprint-Ähnlichkeit (≥ 0.7) | ab 21.11. (braucht GPU) |
| Richtiger Kandidat auf Rang 3–25 | FR 3-25 und P5 (taeyangg4) | vorhanden |

**Validierung:** Der Testdatensatz mit 250 Naturstoffen ist für Rang-1-Fragen ungeeignet, weil Pipeline und FR diese
Moleküle im Training gesehen haben (Rang 1 zu 92–95 % richtig). Deshalb wird auf dem Leaderboard gemessen: immer eine
Änderung pro Abgabe auf derselben Basis (0.495), Bausteine in zwei Dosierungen, damit die Richtung des Effekts sichtbar wird.

**Zeitplan:**
- 19.11.: Baustein A in zwei Dosierungen.
- 21.–27.11.: Baustein B. Erst lokal auf wenigen Molekülen testen, dann ein bis zwei Abgaben.
- bis 07.12.: beste Kombination, Entscheidung über einen Team-Merge (Merge-Frist).
- 07.–14.12.: Finale Auswahl: eine starke und eine robuste Abgabe.

**Grenzen:** 5 Abgaben pro Tag, GPU-Kontingent 6 h pro Woche (Name That LLM teilt es sich). Ein Molekül ist auf dem
Public LB ~0.004 wert, Einzelergebnisse sind deshalb verrauscht. Die Top-10-Teams nutzen nach Angaben aus der Community
keine öffentlichen Notebooks. Gold ist möglich, aber nicht wahrscheinlich, Silber ist der realistische Stand.

## Stand 19.11.2026 (abends)
- Baustein A: GAP 1.5 und GAP 0.75 beide 0.495, genau wie die Kontrolle. Auf den ~125 Public-Molekülen hat der Richter
  entweder kaum getauscht oder Gewinne und Verluste haben sich aufgehoben. Kein messbarer Gewinn; auf dem privaten Teil
  offen. Nicht weiter verfolgen, solange es kein besseres Rang-1-Signal gibt.
- 5.601 Teams, 476 Teams ≥ 0.495 (viele Kopien des taeyangg4-Notebooks). Silbergrenze Rang 280 bei 0.495, eigener
  Rang 175. Gold ab 0.506.
