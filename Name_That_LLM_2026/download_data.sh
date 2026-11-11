#!/usr/bin/env bash
# Lädt die Daten der Kaggle-Competition "name-that-llm" nach ./data
# Voraussetzung: Kaggle-API-Token unter ~/.kaggle/kaggle.json (chmod 600)
# und die Competition-Regeln auf kaggle.com akzeptiert.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
kaggle competitions download -c name-that-llm -p data
cd data
for z in *.zip; do
  [ -e "$z" ] || continue
  unzip -o "$z" && rm "$z"
done
ls -lh
