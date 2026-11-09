"""Match database: seed archive + TennisMyLife yearly files + Matchstat (RapidAPI).

Output: data/generated/matches.csv.gz - one row per completed men's tour-level
main-draw singles match (Grand Slams, ATP Finals, Masters, 500, 250, plus Davis
Cup / team events which are kept for ratings but never bet on).

Cleaning rules (each one fixed a real bug in the previous repository):
* ITF / Challenger / qualifying rows are removed: the archive is main-draw tour
  level only, so mixing them in makes recent form non-comparable.
* The same match reported by two providers (TML dates = tournament start,
  Matchstat dates = match day) is de-duplicated with a +-14 day window and fuzzy
  player names ("Jaume Antoni Munar Clar" == "Jaume Munar").
* Provider level codes are replaced by the tournament registry (Matchstat calls
  most events "A").
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .config import GENERATED, RAW, SEED
from .tournaments import CHALLENGER, EXCLUDED, ITF_TIER, normalize, resolve

MATCHES_PATH = GENERATED / "matches.csv.gz"
SEED_PATH = SEED / "matches_seed.csv.gz"
CHALLENGER_SEED_PATH = SEED / "challenger_seed.csv.gz"   # TennisMyLife Challenger results 2000-2026 (ratings only)
TML_INDEX = "https://stats.tennismylife.org/api/data-files"
TML_GITHUB = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

STAT_COLS = [
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points", "minutes",
]
KEEP_COLS = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "indoor", "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age",
    "score", "best_of", "round", "data_source", *STAT_COLS,
    "w_winners", "w_unforced_errors", "l_winners", "l_unforced_errors",
]


# ---------------------------------------------------------------------------
# Name keys used for fuzzy de-duplication
# ---------------------------------------------------------------------------
def name_keys(value) -> set[str]:
    tokens = normalize(value).split()
    if not tokens:
        return set()
    if len(tokens) == 1:
        return {tokens[0]}
    return {f"{tokens[0][:1]}|{tok}" for tok in tokens[1:] if len(tok) > 1}


def _stat_count(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in STAT_COLS if c in frame.columns]
    return frame[cols].notna().sum(axis=1) if cols else pd.Series(0, index=frame.index)


def dedupe(frame: pd.DataFrame, window_days: int = 14) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    df = frame.reset_index(drop=True).copy()
    dates = pd.to_datetime(pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    src = df.get("data_source", pd.Series("", index=df.index)).astype(str).to_numpy()
    tour = df["tourney_name"].map(lambda n: normalize(n)).to_numpy()
    rnd = df.get("round", pd.Series("", index=df.index)).astype(str).to_numpy()
    stats = _stat_count(df).to_numpy()
    buckets: dict = {}
    for i, (wn, ln) in enumerate(zip(df["winner_name"], df["loser_name"])):
        for wk in name_keys(wn):
            for lk in name_keys(ln):
                buckets.setdefault((wk, lk), []).append(i)
    pairs = set()
    dv = dates.to_numpy()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(set(members), key=lambda i: dv[i] if not pd.isna(dv[i]) else np.datetime64("1900-01-01"))
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if pd.isna(dv[a]) or pd.isna(dv[b]):
                    continue
                if (dv[b] - dv[a]) / np.timedelta64(1, "D") > window_days:
                    break
                pairs.add((min(a, b), max(a, b)))
    drop: set[int] = set()
    fill: dict[int, list[int]] = {}
    for a, b in sorted(pairs):
        if a in drop or b in drop:
            continue
        if src[a] == src[b] and (tour[a] != tour[b] or rnd[a] != rnd[b]):
            continue  # same provider, different round/event = a genuine rematch
        keep, lose = (a, b) if stats[a] >= stats[b] else (b, a)
        drop.add(lose)
        fill.setdefault(keep, []).append(lose)
    for keep, losers in fill.items():
        for lose in losers:
            missing = df.loc[keep].isna()
            cols = missing[missing].index
            if len(cols):
                df.loc[keep, cols] = df.loc[lose, cols].values
    return df.drop(index=list(drop)).reset_index(drop=True), len(drop)


def resolve_level(name, provider_level) -> str:
    lv = str(provider_level or "").strip().upper()
    if CHALLENGER.search(str(name or "")) or lv in {"C", "CH"}:
        return "C"
    if re.search(r"davis cup", str(name), re.I):
        return "D"
    if EXCLUDED.search(str(name or "")):
        return "A"
    if lv in {"G", "M", "F", "500", "250", "D", "O", "C"}:
        return lv  # an explicit archive code is right for that season
    ev = resolve(name)
    if ev is not None:
        return ev.level  # generic "A" (old TML seasons / Matchstat) -> registry
    return lv or "250"


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"input_rows": int(len(frame))}
    df = frame.copy()
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP_COLS]
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date", "winner_name", "loser_name", "surface"])
    df["surface"] = df["surface"].astype(str).str.title()
    df = df[df["surface"].isin(["Hard", "Clay", "Grass", "Carpet"])]
    itf = df["tourney_name"].astype(str).map(lambda n: bool(ITF_TIER.search(n))) | df["tourney_level"].astype(str).str.upper().isin(["S", "F15", "F25"])
    report["lower_tier_removed"] = int(itf.sum())
    df = df[~itf]
    chall = df["tourney_name"].astype(str).map(lambda n: bool(CHALLENGER.search(n)))
    df.loc[chall, "tourney_level"] = "C"
    report["challenger_rows"] = int((df["tourney_level"].astype(str).str.upper() == "C").sum())
    qual = df["round"].astype(str).str.strip().str.match(r"^Q\d?$", case=False)
    report["qualifying_removed"] = int(qual.sum())
    df = df[~qual]
    df["tourney_level"] = [resolve_level(n, lv) for n, lv in zip(df["tourney_name"], df["tourney_level"])]
    # Matchstat rows arrive without a format; men's Slams are best of 5, everything else best of 3.
    bo = pd.to_numeric(df["best_of"], errors="coerce")
    df["best_of"] = bo.where(bo.isin([3, 5]), np.where(df["tourney_level"] == "G", 5, 3)).astype(int)
    for side in ("winner", "loser"):
        df[f"{side}_id"] = df[f"{side}_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    df, removed = dedupe(df)
    report["duplicates_removed"] = int(removed)
    df = df.sort_values(["tourney_date", "tourney_name", "match_num"], na_position="last").reset_index(drop=True)
    report["output_rows"] = int(len(df))
    report["latest_date"] = str(int(df["tourney_date"].max())) if len(df) else None
    return df, report


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def load_seed() -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in (SEED_PATH, CHALLENGER_SEED_PATH) if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_tml_years(years: list[int]) -> tuple[pd.DataFrame, dict]:
    """TennisMyLife yearly files (site index first, GitHub mirror as fallback)."""
    import requests

    status, frames = {}, []
    urls: dict[str, str] = {}
    try:
        r = requests.get(TML_INDEX, timeout=60)
        r.raise_for_status()
        urls = {f["name"]: f["url"] for f in r.json().get("files", [])}
    except Exception as exc:
        status["index_error"] = str(exc)
    jobs = [(y, f"{y}.csv") for y in years] + [(y, f"{y}_challenger.csv") for y in years]
    for year, fname in jobs:
        for url in (urls.get(fname), TML_GITHUB.format(year=year) if fname == f"{year}.csv" else None):
            if not url:
                continue
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                if {"winner_name", "loser_name", "tourney_date"}.issubset(df.columns) and len(df):
                    df["data_source"] = "TennisMyLife"
                    if "challenger" in fname:
                        df["tourney_level"] = "C"
                    (RAW / f"tml_{fname}").write_text(r.text)
                    frames.append(df)
                    status[fname] = {"ok": True, "rows": int(len(df)), "url": url}
                    break
            except Exception as exc:
                status[fname] = {"ok": False, "error": str(exc), "url": url}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), status


def build_database(extra_frames: list[pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict]:
    frames = [load_seed()]
    if MATCHES_PATH.exists():
        frames.append(pd.read_csv(MATCHES_PATH, low_memory=False))
    frames += [f for f in (extra_frames or []) if f is not None and not f.empty]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No match data available (seed missing).")
    merged, report = clean(pd.concat(frames, ignore_index=True, sort=False))
    merged.to_csv(MATCHES_PATH, index=False, compression="gzip")
    return merged, report


def load_matches() -> pd.DataFrame:
    if MATCHES_PATH.exists():
        return pd.read_csv(MATCHES_PATH, low_memory=False)
    df, _ = build_database()
    return df
