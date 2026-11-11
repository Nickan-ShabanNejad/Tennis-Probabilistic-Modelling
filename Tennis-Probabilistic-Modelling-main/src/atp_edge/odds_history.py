"""Historical closing odds (tennis-data.co.uk) - used ONLY for evaluation.

They benchmark the model against the market, fit the market-blend layer and
backtest betting rules. They never enter the feature vector.
Source: http://www.tennis-data.co.uk (check their terms before commercial use).
"""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ODDS

ODDS_PATH = ODDS / "closing_odds_atp.csv.gz"
TENNIS_DATA_URL = "http://www.tennis-data.co.uk/{year}/{year}.xlsx"


def _norm(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z]+", " ", text.lower()).split())


def td_key(name) -> str:
    """'Auger-Aliassime F.' -> 'auger aliassime|f'."""
    name = str(name or "").strip()
    m = re.match(r"^(.*?)\s+((?:[A-Z]\.?-?)+)\.?$", name)
    if not m:
        return _norm(name)
    return _norm(m.group(1)) + "|" + m.group(2)[0].lower()


def full_name_keys(name) -> set[str]:
    tokens = _norm(name).split()
    if len(tokens) < 2:
        return {_norm(name)}
    keys = {" ".join(tokens[k:]) + "|" + tokens[0][0] for k in range(1, len(tokens))}
    keys.add(" ".join(tokens[:-1]) + "|" + tokens[-1][0])
    return keys


def load_odds() -> pd.DataFrame:
    if not ODDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ODDS_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wk"] = df["winner"].map(td_key)
    df["lk"] = df["loser"].map(td_key)
    return df


def attach(frame: pd.DataFrame, odds: pd.DataFrame, winner="_winner", loser="_loser", date="_date") -> pd.DataFrame:
    out = frame.copy()
    cols = ["ps_w", "ps_l", "avg_w", "avg_l", "max_w", "max_l", "w_sets", "l_sets"]
    for c in cols:
        out[c] = np.nan
    if odds.empty or out.empty:
        return out
    idx: dict = {}
    for i, k in enumerate(zip(odds["wk"], odds["lk"])):
        idx.setdefault(k, []).append(i)
    od = odds["date"].to_numpy()
    hit = np.full(len(out), -1)
    for j, (wn, ln, dt) in enumerate(zip(out[winner], out[loser], out[date])):
        d0 = np.datetime64(pd.Timestamp(dt))
        found = -1
        for a in full_name_keys(wn):
            for b in full_name_keys(ln):
                for i in idx.get((a, b), ()):
                    delta = (od[i] - d0) / np.timedelta64(1, "D")
                    if -2 <= delta <= 16:
                        found = i
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        hit[j] = found
    ok = hit >= 0
    for c in cols:
        if c in odds.columns:
            v = np.full(len(out), np.nan)
            v[ok] = pd.to_numeric(odds[c], errors="coerce").to_numpy()[hit[ok]]
            out[c] = v
    return out


def refresh_current_season() -> dict:
    """Best-effort download of this season's tennis-data workbook."""
    import requests

    year = datetime.now(timezone.utc).year
    url = TENNIS_DATA_URL.format(year=year)
    try:
        r = requests.get(url, timeout=120, headers={"User-Agent": "atp-edge"})
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content))

        def col(name):
            return raw[name] if name in raw.columns else pd.Series([np.nan] * len(raw))

        fresh = pd.DataFrame({
            "date": pd.to_datetime(col("Date"), errors="coerce").dt.strftime("%Y-%m-%d"),
            "tournament": col("Tournament"), "series": col("Series"), "surface": col("Surface").astype(str).str.lower(),
            "round": col("Round"), "best_of": col("Best of"), "winner": col("Winner"), "loser": col("Loser"),
            "w_sets": col("Wsets"), "l_sets": col("Lsets"),
            "ps_w": pd.to_numeric(col("PSW"), errors="coerce"), "ps_l": pd.to_numeric(col("PSL"), errors="coerce"),
            "avg_w": pd.to_numeric(col("AvgW"), errors="coerce"), "avg_l": pd.to_numeric(col("AvgL"), errors="coerce"),
            "max_w": pd.to_numeric(col("MaxW"), errors="coerce"), "max_l": pd.to_numeric(col("MaxL"), errors="coerce"),
        }).dropna(subset=["date", "winner", "loser"])
        existing = pd.read_csv(ODDS_PATH) if ODDS_PATH.exists() else pd.DataFrame(columns=fresh.columns)
        current = existing["date"].astype(str).str.startswith(str(year))
        if int(current.sum()) > len(fresh):
            return {"ok": True, "skipped": True, "reason": "bundled file is newer", "url": url}
        merged = pd.concat([existing[~current], fresh], ignore_index=True).sort_values("date")
        merged.to_csv(ODDS_PATH, index=False, compression="gzip")
        return {"ok": True, "rows": int(len(fresh)), "latest": str(fresh["date"].max()), "url": url}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "url": url}
