# Architektur & Datenfluss

Lösung für die Kaggle-Competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). Die Aufgabe: Für einen Text vorhersagen, welches von 18 LLMs ihn geschrieben hat. Bewertet wird mit **Balanced Accuracy**, also dem Mittelwert der Recalls über alle 18 Klassen.

## Überblick

Lokal (auf dem Mac) laufen schnelle Experimente mit gemeinsamer Validierung. Auf Kaggle laufen das große Training und die Abgabe. Dort gibt es kein Internet, das Modell muss also im Notebook trainiert werden oder als Input eingebunden sein.

```mermaid
flowchart LR
    subgraph Quellen
        T[train.csv<br/>2.699 Texte · 150 Prompts]
        P[Open Pool<br/>223.840 Texte<br/>crtproduction · CC BY-SA 4.0]
        X[test.csv<br/>~4.000 versteckte Texte]
    end

    subgraph Lokal["Lokal · src/ + experiments/"]
        F[features.py<br/>TF-IDF + Stilometrie]
        CV[cv.py<br/>GroupKFold nach prompt_id]
        E[Experimente 01–06]
        O[(models/*/oof_*.parquet)]
        F --> E
        CV --> E
        E --> O
    end

    subgraph Kaggle
        B[build_notebook_*.py<br/>bettet src/ ein]
        NB[Notebook<br/>CPU: SVM · GPU: Transformer]
        S[submission.parquet]
        LB[Public / Private LB]
        B --> NB --> S --> LB
    end

    T --> E
    P --> E
    T --> NB
    P --> NB
    X --> NB
    O -. Gewichte fürs Ensemble .-> B
```

## Validierung

Fast jeder Prompt wurde von allen 18 Modellen beantwortet. Wäre derselbe Prompt in Training und Validierung, würde das Modell am Thema statt am Stil lernen. Deshalb sind die Folds nach `prompt_id` gruppiert.

```mermaid
flowchart TB
    D[train.csv] --> G[GroupKFold · 5 Folds · Gruppe = prompt_id]
    G --> OOF[Out-of-Fold-Wahrscheinlichkeiten<br/>je Experiment gespeichert]
    OOF --> M[Balanced Accuracy · Family Accuracy · Log-Loss]
    OOF --> ENS[Ensemble-Gewichte]

    P[Open Pool] --> A[Protokoll A: nur Pool trainieren]
    A --> TA[Test auf der kompletten train.csv]
    TA --> M
```

- **Protokoll A** (nur Pool trainieren, auf der echten train.csv testen): misst, wie gut sich das Gelernte auf echte Wettbewerbstexte überträgt.
- **Protokoll B** (Pool + train.csv-Folds): misst, wie viel die echten Texte zusätzlich bringen und wie stark man sie gewichten sollte.

## Modell 1: LinearSVC auf TF-IDF (aktuelle Abgabe)

```mermaid
flowchart LR
    TXT[Text] --> W[Wort-TF-IDF<br/>1–2-Gramme · max. 300k]
    TXT --> C[Zeichen-TF-IDF<br/>1–5-Gramme · max. 500k<br/>lowercase=False]
    TXT --> ST[Stilometrie<br/>Satzlänge, TTR,<br/>Satzzeichen je 100 Wörter]
    ST --> SC[StandardScaler × 0.1]
    W --> H[hstack · float32 · sparse]
    C --> H
    SC --> H
    H --> SVM[LinearSVC<br/>C=0.1 · class_weight=balanced<br/>train.csv-Texte mit Gewicht 5]
    SVM --> PR[argmax → Modellname]
    SVM --> SM[softmax · T=5<br/>→ Pseudo-Wahrscheinlichkeiten fürs Ensemble]
```

Warum diese Merkmale:

- **Typografie trennt die Familien.** Claude, GLM und Kimi verwenden den Gedankenstrich „—“. GPT, Gemini, Qwen und Kimi verwenden das typografische Apostroph ’. Deshalb bleiben Groß-/Kleinschreibung und Sonderzeichen erhalten.
- **Die Stilometrie wird schwach gewichtet (× 0.1).** Die Genauigkeit bleibt gleich, aber der SVM konvergiert etwa 20-mal schneller als mit voller Gewichtung.
- Code: [`src/features.py`](src/features.py) und [`src/linear_large.py`](src/linear_large.py). Das Notebook baut [`kaggle/build_notebook_svm.py`](kaggle/build_notebook_svm.py).

## Modell 2: ModernBERT auf dem Pool (in Arbeit)

```mermaid
flowchart LR
    P[Open Pool] --> TK[Tokenizer · max. 384 Tokens]
    TK --> MB[ModernBERT-base<br/>Fine-Tuning auf Kaggle-GPU<br/>fp16 · gewichtete Cross-Entropy]
    MB --> W[(Gewichte als<br/>Notebook-Output)]
    MB --> TP[Wahrscheinlichkeiten<br/>auf train.csv]
    W --> INF[Abgabe-Notebook<br/>nur Inferenz]
```

Das Training läuft in einem eigenen Notebook mit Internet, weil es ModernBERT von Hugging Face lädt. Das Abgabe-Notebook bindet dessen Output als Input ein und läuft ohne Internet. Code: [`kaggle/build_notebook_bert.py`](kaggle/build_notebook_bert.py).

## Ensemble (geplant)

```mermaid
flowchart LR
    S[SVM-Wahrscheinlichkeiten] --> B[Gewichteter Mittelwert<br/>Gewichte aus den Ergebnissen auf train.csv]
    T[Transformer-Wahrscheinlichkeiten] --> B
    B --> A[argmax] --> SUB[submission.parquet]
```

TF-IDF erkennt Typografie und Wortwahl, der Transformer eher Stil und Struktur. Mit 3.000 Pool-Texten pro Modell, getestet auf train.csv:

- SVM allein: 0.449
- ModernBERT allein: 0.479
- Mischung: 0.508–0.509

Die Mischung ist robust: Der Wert gilt für viele Kombinationen, etwa Softmax-Temperatur 5 mit SVM-Gewicht 0.4–0.6 oder Temperatur 10 mit SVM-Gewicht 0.3–0.4.

Nach dem Mischen gleicht ein **Sinkhorn-Ausgleich** (Stärke 0.6) die vorhergesagten Klassenanteile an. In einer Simulation auf train.csv brachte er +1,3 Punkte bei ausgewogenem Test und +0,8 bei stark unausgewogenem. Bei einem Modell, das nach Stufe 2 auf der ausgewogenen train.csv trainiert ist, ist der Gewinn kleiner. Das Abgabe-Notebook baut [`kaggle/build_notebook_ensemble.py`](kaggle/build_notebook_ensemble.py).

## Ergebnisse

| Ansatz | Trainingsdaten | Balanced Accuracy | Public LB |
|---|---|---|---|
| TF-IDF + LogReg + Stilometrie | train.csv | 0.352 (CV) | 0.320 |
| Hierarchisch (Familie → Modell) | train.csv | 0.351 (CV) | – |
| ModernBERT-base | train.csv, Fold 0 | 0.357 | – |
| LinearSVC | Pool, 1.000 / Modell | 0.400 (Protokoll A) | – |
| LinearSVC | Pool, 3.000 / Modell | 0.449 (Protokoll A) | – |
| ModernBERT-base, 2 Epochen | Pool, 3.000 / Modell | 0.479 (Protokoll A) | – |
| Mischung SVM + ModernBERT (T=5, 50/50) | Pool, 3.000 / Modell | 0.508 (Protokoll A) | – |
| ModernBERT Stufe 1 + Stufe 2 (2 Epochen train.csv) | Pool 3.000 / Modell, dann train.csv | 0.518 (5-Fold-OOF) | – |
| Mischung SVM + ModernBERT Stufe 2 (T=5, 50/50) | wie oben | 0.530 (OOF) | – |
| LinearSVC | Pool, 6.000 / Modell | 0.489 (Protokoll A) | – |
| LinearSVC | kompletter Pool + train.csv | 0.502 (OOF) | 0.470 |
| ModernBERT Stufe 1 + 2 | kompletter Pool, dann train.csv (2 Ep.) | 0.583 (OOF) | – |
| Qwen3-4B-LoRA Stufe 1 + 2 | Pool-Tripel, dann train.csv | 0.525 (OOF) | – |
| **Mischung SVM 0.4 + ModernBERT 0.6 (T=5)** | wie oben | 0.596 (OOF, ehrlich) | **0.565** |
| Mischung SVM 0.4 + ModernBERT 0.4 + LoRA 0.2 (T=5) | wie oben | 0.603 (OOF, ehrlich) | ausstehend |

Klassen-Bias (`src/postprocess.py`) auf der Mischung: −1,0 bis −1,2 Pp. in der CV, wird nicht verwendet.
Abstand OOF → Public LB bisher etwa 3 Pp. Öffentliche Notebooks berichten, dass Sinkhorn, Zentrierung und Hungarian-Zuordnung
auf dem LB verlieren (vermutlich wenige Texte je Prompt im Test), daher Abgabe ohne Sinkhorn.

## Abgaben auf Basis öffentlicher Notebooks (ab 14.11.)

Statt die Community-Modelle nachzubauen, laufen die Abgaben seit dem 14.11. auf öffentlichen Notebooks (alle Bausteine
öffentlich, Credits im Notebook). Unser eigener Beitrag ist der **GLM-Likelihood-Klassifikator aus
[Experiment 08](experiments/08_glm_likelihood/README.md)**: Gradient Boosting auf train.csv + 5.670 selbst unter
GLM-5.3 / GLM-5.3-Flash bewerteten Open-Pool-Texten statt der LogReg nur auf train.csv.

| Abgabe | Basis | Änderung | Public LB |
|---|---|---|---|
| V1 | 0.694 (lightningv08) | GLM-Stufe: unsichere Texte auswählen (glm, glm-flash, fable), Abdeckung 0.30 | kein Score |
| V2 / V2b | 0.694 | GSQ-27B-Anteil 0.12 → 0.03 | kein Score |
| V3 / V3b | 0.694 | V1 + V2 | kein Score |
| V4 | 0.694 | GSQ-27B-Stufe aus, damit das 9-h-Limit hält | 0.685 |
| V5 | V4 | GLM-Klassifikator: HGB auf train + Pool (Pool-Gewicht 0.5, W = 1.0) | 0.695 (+1,0 ggü. V4) |
| **V6** | 0.701 (alexchilton, Linie lumibee Kiro v44) | wie V5, sonst unverändert | **0.707** |
| V7 | 0.702 (lightningv08 Soft Typography) | wie V5, sonst unverändert | 0.704 |
| V8 | V7 | Apex-Nachbearbeitung auf den Logits nach der GLM-Stufe ([Experiment 09](experiments/09_postprocess/README.md), +0,4 in der Simulation) | – |

V1–V3b laufen auf Kaggle durch, bekommen aber keinen Score (vermutlich 9-h-Limit im Bewertungslauf, daher V4 ohne GSQ).
Code: [`kaggle/public_694/`](kaggle/public_694/README.md) (V1–V5), [`kaggle/public_701/`](kaggle/public_701/README.md) (V6)
und [`kaggle/public_702/`](kaggle/public_702/README.md) (V7).

**Stand 17.11.2026:** Platz 115 von 6.165 Teams (Public). Grenzen: Bronze (Top 10 %) 0.698, Silber (Top 5 %, Platz 308) 0.701,
Gold (Platz 22) 0.743, Platz 10 0.758, Platz 1 0.776. Bestes öffentliches Notebook: 0.702.

## Ordnerstruktur

```
Name_That_LLM_2026/
├── src/                  gemeinsamer Code (Daten, Features, CV, Modelle)
├── experiments/          ein Ordner je Ansatz, mit README und Skripten
│   ├── 01_tfidf_linear/
│   ├── 02_stylometry/
│   ├── 03_hierarchical/
│   ├── 04_transformer/   ModernBERT lokal (MPS)
│   ├── 05_ensemble/
│   ├── 06_external_data/ Open Pool: Skalierung & Kombination
│   ├── 08_glm_likelihood/ GLM-Likelihood-Klassifikator mit Pool-Daten (V5/V6)
│   └── 09_postprocess/   GLM-Stufe vs. Apex-Nachbearbeitung im 0.702-Notebook (V8)
├── kaggle/               Notebook-Generatoren (build_notebook_*.py) + Kaggle-Metadaten
│   ├── svm_pool/         Abgabe: LinearSVC (CPU, offline)
│   ├── svm_oof/          OOF der SVM auf train.csv (CPU) für Ensemble-Gewichte
│   ├── bert_pool/        Training Stufe 1: ModernBERT auf dem Pool (GPU)
│   ├── bert_stage2/      Training Stufe 2: Feintuning auf train.csv + CV (GPU)
│   ├── lora/             LoRA auf Qwen3 (4B/8B), zweistufig, eigene LoRA-Schicht ohne peft (GPU)
│   ├── ensemble/         Abgabe: SVM + ModernBERT Stufe 2 + Sinkhorn (GPU, offline)
│   ├── glm_pool/         Pool-Texte unter GLM-5.3 / GLM-5.3-Flash bewerten (GPU, Experiment 08)
│   ├── public_694/       Fork des 0.694-Notebooks, Varianten V1–V5
│   ├── public_701/       Fork des 0.701-Notebooks, V6 (aktuell beste Abgabe, 0.707)
│   └── public_702/       Fork des 0.702-Notebooks (Soft Typography), V7
├── notebooks/01_EDA.ipynb
├── data/                 nicht im Git (download_data.sh)
└── models/               OOF-Vorhersagen, nicht im Git
```
