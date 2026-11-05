"""Manual injury flags (data/reference/injuries.csv).

Automatic injury signals (retirements, walkovers, layoffs, comeback matches) are
learned by the model from match history. News-based injuries are not available
from any reliable free API, so they are entered here and applied as a judgment
adjustment to the final probability (see betting.injury_adjust).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import REFERENCE
from .tournaments import normalize

PATH = REFERENCE / "injuries.csv"


def load(today: date | None = None) -> dict[str, dict]:
    if not PATH.exists():
        return {}
    df = pd.read_csv(PATH, comment="#", skip_blank_lines=True)
    today = today or date.today()
    out = {}
    for row in df.itertuples(index=False):
        name = str(getattr(row, "player", "") or "").strip()
        if not name:
            continue
        until = pd.to_datetime(getattr(row, "until", None), errors="coerce")
        if pd.notna(until) and until.date() < today:
            continue
        try:
            sev = float(getattr(row, "severity", 0) or 0)
        except Exception:
            sev = 0.0
        out[normalize(name)] = {"player": name, "severity": max(0.0, min(1.0, sev)), "note": str(getattr(row, "note", "") or "")}
    return out


def severity(name: str, flags: dict) -> float:
    return float(flags.get(normalize(name), {}).get("severity", 0.0))
