"""Schlanker Kaggle-Client (ohne kaggle-CLI) für Download, Submit, Limits und Scores.

Nutzt die RPC-API (`/api/v1/<Service>/<Methode>`, wie das offizielle kagglesdk).
Auth: KAGGLE_API_TOKEN bzw. ~/.kaggle/access_token (Bearer) oder
KAGGLE_USERNAME/KAGGLE_KEY bzw. ~/.kaggle/kaggle.json (Basic). In der Claude-Cloud-Umgebung
injiziert der Proxy die Zugangsdaten automatisch – dann geht es ganz ohne Key.

    python -m src.kaggle_api download
    python -m src.kaggle_api limits
    python -m src.kaggle_api submissions
    python -m src.kaggle_api leaderboard
    python -m src.kaggle_api submit submissions/01_lgbm_baseline.csv "LGBM baseline | CV 0.94631"
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
COMP_SVC = "competitions.CompetitionApiService"


def _session() -> requests.Session:
    s = requests.Session()
    token = os.getenv("KAGGLE_API_TOKEN")
    tfile = Path.home() / ".kaggle" / "access_token"
    if not token and tfile.exists():
        token = tfile.read_text().strip()
    user, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cfg.exists():
        c = json.loads(cfg.read_text())
        user, key = c.get("username"), c.get("key")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    elif user and key:
        s.auth = (user, key)
    return s


def _rpc(method: str, body: dict, service: str = COMP_SVC) -> dict:
    r = _session().post(f"{BASE}/{service}/{method}", json=body)
    if not r.ok:
        raise RuntimeError(f"{method}: HTTP {r.status_code} {r.text[:300]}")
    return r.json() if r.text else {}


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


def limits() -> dict:
    res = _rpc("GetSubmissionLimits", {"competitionName": COMPETITION})
    print("heute noch erlaubt:", res.get("numAllowedNow"))
    return res


def submissions(n: int = 10, quiet: bool = False) -> list[dict]:
    rows = _rpc("ListSubmissions", {"competitionName": COMPETITION}).get("submissions", [])[:n]
    if not quiet:
        for s in rows:
            print(f"{str(s.get('date', ''))[:19]}  {s.get('status', 'PENDING'):<10} "
                  f"public={s.get('publicScore', '-')!s:<9} {s.get('fileName', '')}  "
                  f"| {s.get('description', '')}")
    return rows


def leaderboard(my_score: float | None = None, quiet: bool = False) -> dict:
    """Ganzes Public-LB laden; Quantile, eigener Rang und Rang eines hypothetischen Scores."""
    import io

    import pandas as pd

    r = _session().get(f"{BASE}/competitions/{COMPETITION}/leaderboard/download")
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        lb = pd.read_csv(z.open(z.namelist()[0]))
    s = lb["Score"].sort_values(ascending=False).reset_index(drop=True)
    best = [x for x in submissions(50, quiet=True) if x.get("publicScore")]
    mine = max((float(x["publicScore"]) for x in best), default=None)
    out = {"teams": len(s), "top1": s.iloc[0], "top10": s.iloc[min(9, len(s) - 1)],
           "top50": s.iloc[min(49, len(s) - 1)], "median": s.median(), "my_best": mine,
           "my_rank": int((s > mine).sum() + 1) if mine is not None else None}
    if my_score is not None:
        out["rank_if"] = int((s > my_score).sum() + 1)
    if not quiet:
        print(f"LB: {out['teams']} Teams · #1 {out['top1']:.5f} · #10 {out['top10']:.5f} · "
              f"#50 {out['top50']:.5f} · Median {out['median']:.5f}")
        if mine is not None:
            print(f"Wir: {mine:.5f} → Rang {out['my_rank']}/{out['teams']} "
                  f"(Top {100 * out['my_rank'] / out['teams']:.0f} %), Abstand zu #1 {out['top1'] - mine:+.5f}")
        if my_score is not None:
            print(f"Mit {my_score:.5f} wären wir auf Rang {out['rank_if']}")
    return out


def submit(path: str, message: str, wait: bool = True) -> dict | None:
    p = Path(path)
    # 1) Upload-URL holen
    up = _rpc("StartSubmissionUpload", {
        "competitionName": COMPETITION, "fileName": p.name,
        "contentLength": p.stat().st_size, "lastModifiedEpochSeconds": int(p.stat().st_mtime)})
    # 2) Datei hochladen (signierte URL, keine Kaggle-Auth)
    with p.open("rb") as f:
        r = requests.put(up["createUrl"], data=f)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Upload fehlgeschlagen: HTTP {r.status_code} {r.text[:300]}")
    # 3) Submission anlegen
    res = _rpc("CreateSubmission", {"competitionName": COMPETITION,
                                    "blobFileTokens": up["token"], "submissionDescription": message})
    print("eingereicht:", res.get("message", res))
    if not wait:
        return None
    for _ in range(60):  # bis zu 10 Min. auf den Score warten
        time.sleep(10)
        last = submissions(1, quiet=True)
        status = str(last[0].get("status", "PENDING")).upper() if last else "PENDING"
        if status not in ("PENDING", ""):  # leerer Status = Default PENDING
            submissions(1)
            return last[0]
    print("Score noch ausstehend – später `python -m src.kaggle_api submissions`")
    return None


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["submissions"]
    if cmd == "download":
        download()
    elif cmd == "limits":
        limits()
    elif cmd == "submissions":
        submissions()
    elif cmd == "leaderboard":
        leaderboard(float(args[0]) if args else None)
    elif cmd == "submit":
        submit(args[0], args[1] if len(args) > 1 else Path(args[0]).stem)
    else:
        sys.exit(f"unbekannter Befehl: {cmd} (download | limits | submissions | leaderboard [score] | submit <csv> [msg])")
