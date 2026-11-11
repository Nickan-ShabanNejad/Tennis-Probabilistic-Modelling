"""Closing lines of matches that have ALREADY been played, as an input to the ratings.

The betting market knows things a results-only rating can't: a player carrying an
injury, a youngster who is much better than his record, a veteran who is finished.
How the market priced a player in his recent matches, compared with what his Elo
said, is therefore useful information about his next match.

This module attaches ``mkt_p_winner`` (the closing no-vig probability of the eventual
winner) to the match table. The engine turns it into the ``mkt_offset`` feature.
Only finished matches are used, so there is no look-ahead: today's price never
enters today's prediction through this route.

Two sources, best first:
1. the closing-odds archive (tennis-data.co.uk): Pinnacle, else the books' average;
2. this repository's own prediction log: the last pre-match Pinnacle price (or the
   books' consensus) recorded by the live job - this keeps the feature current even
   when the archive is weeks behind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .betting import no_vig_array
from .config import PREDICTIONS
from .odds_history import attach, load_odds
from .tournaments import BETTABLE_LEVELS, normalize

PRED_LOG = PREDICTIONS / "predictions_log.csv"


def _pkey(name) -> str:
    t = normalize(name).split()
    return (t[0][:1] if t else "") + "|" + (t[-1] if t else "")


def _from_log(matches: pd.DataFrame, dates: pd.Series, already: np.ndarray) -> np.ndarray:
    """Winner's closing probability from the live prediction log (NaN where not found)."""
    out = np.full(len(matches), np.nan)
    if not PRED_LOG.exists():
        return out
    try:
        log = pd.read_csv(PRED_LOG)
    except Exception:
        return out
    need = {"player_a", "player_b", "start_ts", "p_market_a"}
    if log.empty or not need.issubset(log.columns):
        return out
    log = log.dropna(subset=["start_ts", "p_market_a"])
    if log.empty:
        return out
    pin = no_vig_array(pd.to_numeric(log.get("pinnacle_a"), errors="coerce"), pd.to_numeric(log.get("pinnacle_b"), errors="coerce")) \
        if {"pinnacle_a", "pinnacle_b"}.issubset(log.columns) else np.full(len(log), np.nan)
    p_a = np.where(np.isfinite(pin), pin, pd.to_numeric(log["p_market_a"], errors="coerce").to_numpy(float))
    start = pd.to_datetime(log["start_ts"], unit="s", errors="coerce").dt.normalize()
    index: dict = {}
    for a, b, d, p in zip(log["player_a"], log["player_b"], start, p_a):
        if pd.isna(d) or not np.isfinite(p):
            continue
        ka, kb = _pkey(a), _pkey(b)
        index.setdefault(tuple(sorted((ka, kb))), []).append((d, ka, float(p)))
    if not index:
        return out
    first = min(d for rows in index.values() for d, _, _ in rows) - pd.Timedelta(days=20)
    recent = np.flatnonzero((dates >= first).to_numpy() & ~already)
    for i in recent:
        wk, lk = _pkey(matches["winner_name"].iat[i]), _pkey(matches["loser_name"].iat[i])
        for d, ka, p in index.get(tuple(sorted((wk, lk))), ()):
            # archive dates are the tournament start (or the match day); the log has the match day
            if -3 <= (d - dates.iat[i]).days <= 16:
                out[i] = p if ka == wk else 1 - p
                break
    return out


def attach_market(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return ``matches`` with a ``mkt_p_winner`` column and a small coverage report."""
    out = matches.copy()
    out["mkt_p_winner"] = np.nan
    if out.empty:
        return out, {"priced_matches": 0}
    dates = pd.to_datetime(pd.to_numeric(out["tourney_date"], errors="coerce").astype("Int64").astype(str),
                           format="%Y%m%d", errors="coerce")
    tour = out["tourney_level"].astype(str).isin(BETTABLE_LEVELS).to_numpy() & dates.notna().to_numpy()
    p = np.full(len(out), np.nan)
    odds = load_odds()
    if not odds.empty and tour.any():
        sub = out.loc[tour].assign(_d=dates[tour])
        att = attach(sub, odds, winner="winner_name", loser="loser_name", date="_d")
        pin = no_vig_array(att["ps_w"], att["ps_l"])
        avg = no_vig_array(att["avg_w"], att["avg_l"])
        p[tour] = np.where(np.isfinite(pin), pin, avg)
    from_archive = int(np.isfinite(p).sum())
    live = _from_log(out, dates, np.isfinite(p))
    p = np.where(np.isfinite(p), p, live)
    out["mkt_p_winner"] = p
    have = np.isfinite(p)
    latest = dates[have].max() if have.any() else None
    return out, {"priced_matches": int(have.sum()), "from_archive": from_archive, "from_live_log": int(have.sum()) - from_archive,
                 "latest_priced_match": None if latest is None or pd.isna(latest) else str(latest.date())}
