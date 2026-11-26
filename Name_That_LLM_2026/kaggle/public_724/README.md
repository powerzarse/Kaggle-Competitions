# Fork: 0.724 | PB-115 Likelihood & Unlocked GLM Coverage (lumibee v69)

Quelle: https://www.kaggle.com/code/lumibee/0-724-pb-115-likelihood-unlocked-glm-coverage (am 25.11. per
`kaggle kernels pull -m` geholt, `public_724.ipynb` + `original-kernel-metadata.json`; Credits im Notebook).

Gegenüber unserer alten Basis (alexchilton 0.701 → V6 0.707) neu: GLM-Kopf „PB-115“ (54 Zusammenfassungen + 61
token-gleiche Merkmale GLM-5.3 gegen Flash, 0.55 LR + 0.45 · 3 LGB, nur train.csv, Datensatz
`lumibee/ntl-prefixb-full-115`), Prefix B (Nutzer-Turn je Text, 160 Tokens), Hybrid aus 18-Klassen-log-p und marginalen
GLM-log-odds (α 0.35), CompStack aus → GLM-Abdeckung bis 40 %, Guarded Fable Return, Sibling-Residual in Apex,
gemeinsame Typografie-Maske. Details: `experiments/17_pb115_pool/README.md`.

## V21 (`v21_fork/`, Kaggle: `linuszarse/ntl-724-fork-v21`)
Unverändert (Notebook bytegleich mit `public_724.ipynb`), nur ID/Titel/privat in `kernel-metadata.json`
(Docker-Image von lumibee übernommen). Commit-Lauf 25.11. OK (FAST_COMMIT, „V1 checks OK“, „PB115 … smoke ok“;
„Commit diagnostic skipped: NameError Path“ ist lumibees eigene Commit-Diagnose in `try`, auch im Original).
Abgabe 25.11. 12:35, Ref 58884402. Hinweis: Der Open Pool ist seit 25.11. 07:34 in einer neuen Version (239k statt
224k Texte); V21 nutzt die neue, lumibees gewertete Version vermutlich die alte.

**Public LB: 0.724** (gewertet 25.11. ~19:50) = lumibee → reproduziert trotz neuer Pool-Version. Platz 135/7.089
(Silberzone; Bronze 0.720, Silber 0.721, Gold ~0.759, Platz 1 0.813). Neue Basis statt V6 (0.707).

## V22 (`v22_pbpool/`, `build_v22_pbpool.py`, Kaggle: `linuszarse/ntl-724-fork-v22-pbpool`)
V21 + PB-115-Kopf auf train + Prefix-B-Pool (privater Datensatz `linuszarse/ntl-pb115-pool`, klassenweise
ausgeglichene Gewichte, Mittel 0.5). Ohne Datensatz bitgleich V21 (lokal getestet), mit Datensatz lokal Ende-zu-Ende
getestet (4.734 Pool-Zeilen). **Verworfen, nicht gepusht/abgegeben:** Simulation auf echten Prefix-B-Daten −0,34 Pp.
[−0,63; −0,04] (alle Pool-Varianten −0,1 bis −0,4), siehe `experiments/17_pb115_pool/README.md`.

Push: `.venv/bin/kaggle kernels push -p kaggle/public_724/v21_fork` (T4, GPU-Kontingent; Commit ~1 h inkl. Warteschlange)
