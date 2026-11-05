"""Court Speed Index (CSI).

Primary source: Tennis Abstract surface-speed ratings (Jeff Sackmann). They are
derived from ace rate adjusted for the servers and returners in each match, so
they capture surface, balls, altitude and weather. 1.00 = tour average,
1.25 = 25% more aces than average. https://www.tennisabstract.com/reports/atp_surface_speed.html

Secondary source: our own estimate from match statistics (same idea: aces per
service point relative to what those players normally produce), used when an
event is missing from Tennis Abstract and to update the rating DURING an event
from matches already played that week.

Leakage rule: a match is only ever given ratings from earlier seasons plus earlier
matches of the same event.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REFERENCE
from .tournaments import EVENTS, canonical, resolve

TA_CURRENT = "https://www.tennisabstract.com/reports/atp_surface_speed.html"
TA_YEAR = "https://www.tennisabstract.com/cgi-bin/surface-speed.cgi?year={year}"
TA_PATH = REFERENCE / "court_speed_tennis_abstract.csv"
SURFACE_DEFAULT = {"Hard": 1.05, "Clay": 0.80, "Grass": 1.25, "Carpet": 1.2}

_ROW = re.compile(
    r"<tr><td[^>]*>([^<]+)</td><td[^>]*>(?:<a[^>]*>)?([^<]+)(?:</a>)?</td><td[^>]*>([^<]+)</td>"
    r"<td[^>]*>([\d.]+)%</td><td[^>]*>([\d.]+)</td></tr>"
)


def parse_tennis_abstract(html: str) -> pd.DataFrame:
    rows = []
    for date, name, surface, ace, speed in _ROW.findall(html):
        parsed = pd.to_datetime(date.strip(), errors="coerce", dayfirst=False)
        if pd.isna(parsed):
            continue
        rows.append({
            "date": parsed.strftime("%Y-%m-%d"), "year": int(parsed.year), "tournament": name.strip(),
            "key": canonical(name), "surface": surface.strip().title(), "ace_pct": float(ace) / 100.0,
            "speed": float(speed), "source": "tennis_abstract",
        })
    return pd.DataFrame(rows)


def fetch_tennis_abstract(years: list[int] | None = None, session=None, pause: float = 1.0) -> pd.DataFrame:
    """Download ratings. ``years`` = historical seasons; the current 52 weeks are always fetched."""
    import requests

    s = session or requests.Session()
    headers = {"User-Agent": "atp-edge research model (court speed index; weekly refresh)"}
    frames = []
    urls = [TA_CURRENT] + [TA_YEAR.format(year=y) for y in (years or [])]
    for url in urls:
        for attempt in range(4):
            try:
                r = s.get(url, timeout=60, headers=headers)
                if r.status_code == 429:  # be polite: back off and retry
                    time.sleep(30 * (attempt + 1))
                    continue
                r.raise_for_status()
                frames.append(parse_tennis_abstract(r.text))
                break
            except Exception as exc:  # keep going; the table is cumulative
                print(f"WARNING court speed fetch failed {url}: {exc}")
                break
        time.sleep(pause)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_into_table(fresh: pd.DataFrame, path: Path = TA_PATH) -> pd.DataFrame:
    old = pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    both = pd.concat([old, fresh], ignore_index=True) if not old.empty else fresh
    if both.empty:
        return both
    both["key"] = both["tournament"].map(canonical)
    both = both.drop_duplicates(["year", "key", "surface"], keep="last").sort_values(["year", "date"])
    both.to_csv(path, index=False)
    return both


class CourtSpeed:
    """Leakage-safe lookups of the court speed index for (event, surface, season)."""

    def __init__(self, table: pd.DataFrame | None = None):
        if table is None:
            table = pd.read_csv(TA_PATH) if TA_PATH.exists() else pd.DataFrame()
        self.table = table
        self._by_event: dict[tuple[str, str], list[tuple[int, float]]] = {}
        if not table.empty:
            for row in table.itertuples(index=False):
                self._by_event.setdefault((str(row.key), str(row.surface)), []).append((int(row.year), float(row.speed)))
            for k in self._by_event:
                self._by_event[k].sort()
        surf = table.groupby("surface")["speed"].median().to_dict() if not table.empty else {}
        self.surface_default = {**SURFACE_DEFAULT, **{k: float(v) for k, v in surf.items()}}

    def prior(self, tournament: str, surface: str, season: int) -> tuple[float, int]:
        """Recency-weighted mean of earlier seasons (weights 1, .6, .36 ...).

        Returns (speed, n_seasons). n_seasons = 0 means the surface default.
        """
        key = canonical(tournament)
        hist = [(y, v) for y, v in self._by_event.get((key, surface), []) if y < season]
        if not hist:
            return self.surface_default.get(surface, 1.0), 0
        hist = hist[-4:]
        w = np.array([0.6 ** (season - 1 - y) for y, _ in hist])
        v = np.array([v for _, v in hist])
        est = float((w * v).sum() / w.sum())
        # shrink lightly toward the surface default when only one season exists
        n = len(hist)
        base = self.surface_default.get(surface, 1.0)
        return float((n * est + 0.5 * base) / (n + 0.5)), n

    def current_or_prior(self, tournament: str, surface: str, season: int) -> tuple[float, str]:
        """For live predictions: this season's rating if the event already has one
        (e.g. a later round of an event TA has rated), otherwise the prior."""
        key = canonical(tournament)
        now = [v for y, v in self._by_event.get((key, surface), []) if y == season]
        if now:
            return float(now[-1]), "tennis_abstract_current"
        speed, n = self.prior(tournament, surface, season)
        return speed, ("tennis_abstract_history" if n else "surface_default")


def blend_live(prior: float, live: float | None, live_matches: int, k: float = 12.0) -> tuple[float, float]:
    """Shrink the in-event estimate toward the prior. Returns (speed, live_weight)."""
    if live is None or live_matches <= 0:
        return float(prior), 0.0
    w = live_matches / (live_matches + k)
    return float((1 - w) * prior + w * live), float(w)


def speed_bucket(speed: float) -> str:
    if speed < 0.80:
        return "slow"
    if speed < 0.95:
        return "medium-slow"
    if speed < 1.10:
        return "medium"
    if speed < 1.25:
        return "medium-fast"
    return "fast"


def event_surface(tournament: str, default: str = "Hard") -> str:
    ev = resolve(tournament)
    return ev.surface if ev else default


__all__ = ["CourtSpeed", "fetch_tennis_abstract", "merge_into_table", "blend_live", "speed_bucket", "EVENTS"]
