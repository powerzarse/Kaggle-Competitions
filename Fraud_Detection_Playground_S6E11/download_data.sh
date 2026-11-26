#!/usr/bin/env bash
# Lädt die Daten von playground-series-s6e11 nach ./data (siehe src/kaggle_api.py für Auth).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.kaggle_api download
ls -lh data
