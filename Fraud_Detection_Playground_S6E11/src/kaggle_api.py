"""Schlanker Kaggle-REST-Client (ohne kaggle-CLI) für Download, Submit und Score-Abfrage.

Auth: KAGGLE_USERNAME/KAGGLE_KEY bzw. ~/.kaggle/kaggle.json (Basic-Auth) oder
KAGGLE_API_TOKEN bzw. ~/.kaggle/access_token (Bearer). In der Claude-Cloud-Umgebung
injiziert der Proxy die Zugangsdaten automatisch – dann geht es ganz ohne Key.

    python -m src.kaggle_api download
    python -m src.kaggle_api submit submissions/01_lgbm_baseline.csv "LGBM baseline, CV 0.9xx"
    python -m src.kaggle_api submissions
"""
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

from .config import COMPETITION, DATA_DIR

BASE = "https://www.kaggle.com/api/v1"


def _session() -> requests.Session:
    s = requests.Session()
    user, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cfg.exists():
        c = json.loads(cfg.read_text())
        user, key = c.get("username"), c.get("key")
    token = os.getenv("KAGGLE_API_TOKEN")
    tfile = Path.home() / ".kaggle" / "access_token"
    if not token and tfile.exists():
        token = tfile.read_text().strip()
    if user and key:
        s.auth = (user, key)
    elif token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def download(dest: Path = DATA_DIR) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    r = _session().get(f"{BASE}/competitions/data/download-all/{COMPETITION}", stream=True)
    r.raise_for_status()
    zpath = dest / f"{COMPETITION}.zip"
    with zpath.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink()
    print("entpackt nach", dest, sorted(p.name for p in dest.glob("*.csv")))


def submissions(n: int = 10) -> list[dict]:
    r = _session().get(f"{BASE}/competitions/submissions/list/{COMPETITION}")
    r.raise_for_status()
    rows = r.json()[:n]
    for s in rows:
        print(f"{s.get('date', '')[:19]}  {s.get('status', ''):<10} "
              f"public={s.get('publicScore', '')!s:<10} {s.get('fileName', '')}  "
              f"| {s.get('description', '')}")
    return rows


def submit(path: str, message: str, wait: bool = True) -> None:
    p = Path(path)
    s = _session()
    # 1) Upload-URL anfordern
    r = s.post(f"{BASE}/competitions/{COMPETITION}/submissions/url",
               data={"fileName": p.name, "contentLength": p.stat().st_size,
                     "lastModifiedDateUtc": int(p.stat().st_mtime)})
    r.raise_for_status()
    info = r.json()
    # 2) Datei hochladen
    with p.open("rb") as f:
        up = requests.put(info["createUrl"], data=f)
    up.raise_for_status()
    # 3) Submission auslösen
    r = s.post(f"{BASE}/competitions/submissions/submit/{COMPETITION}",
               data={"blobFileTokens": info["token"], "submissionDescription": message})
    r.raise_for_status()
    print("eingereicht:", r.json().get("message", r.text[:200]))
    if wait:
        for _ in range(30):
            time.sleep(10)
            last = submissions(1)
            if last and last[0].get("status", "").lower() not in ("pending", "submissionstatus.pending"):
                break


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["submissions"]
    if cmd == "download":
        download()
    elif cmd == "submissions":
        submissions()
    elif cmd == "submit":
        submit(args[0], args[1] if len(args) > 1 else Path(args[0]).stem)
    else:
        sys.exit(f"unbekannter Befehl: {cmd} (download | submissions | submit <csv> [msg])")
