"""Matchstat "Tennis API - ATP WTA ITF" on RapidAPI.

Used for: upcoming ATP fixtures, recent completed results (with match stats),
current rankings, and (optionally) bookmaker odds for the total-sets market.
Key: RAPIDAPI_KEY (MATCHSTAT_API_KEY also accepted).

Response shapes differ between Matchstat endpoints, so every parser is tolerant
and keeps missing statistics as NaN (never zero).
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from .config import secret
from .tournaments import normalize

HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
BASE = f"https://{HOST}"


def payload_rows(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "singles", "results", "events", "matches", "items", "result", "fixtures"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = payload_rows(value)
            if nested:
                return nested
    return []


def _name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for k in ("name", "fullName", "displayName", "playerName"):
            if value.get(k):
                return str(value[k]).strip()
    return ""


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _ts(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value) / (1000.0 if value > 1e11 else 1.0)
    try:
        return float(pd.to_datetime(value, utc=True).timestamp())
    except Exception:
        return None


class MatchstatClient:
    def __init__(self, api_key: str | None = None, min_interval: float = 0.7, timeout: float = 45.0):
        self.api_key = (api_key or secret("RAPIDAPI_KEY")).strip()
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY is not set")
        self.s = requests.Session()
        self.s.headers.update({"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": HOST, "User-Agent": "atp-edge/1.0"})
        self.min_interval, self.timeout = min_interval, timeout
        self._last = 0.0
        self.calls = 0

    def get(self, path: str, params: dict | None = None) -> Any:
        last_err = None
        for attempt in range(4):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(f"{BASE}/{path.lstrip('/')}", params=params or {}, timeout=self.timeout)
                self._last = time.monotonic()
                self.calls += 1
                if r.status_code == 429:
                    time.sleep(min(float(r.headers.get("Retry-After", 3 + 3 * attempt)), 30))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                last_err = exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Matchstat request failed {path}: {last_err}")

    def first_working(self, paths: Iterable[str], params: dict | None = None) -> Any:
        errors = []
        for p in paths:
            try:
                return self.get(p, params)
            except Exception as exc:
                errors.append(str(exc))
        raise RuntimeError(" | ".join(errors))

    # ------------------------------------------------------------- rankings
    def rankings(self, n: int = 500) -> pd.DataFrame:
        rows, page = [], 1
        while len(rows) < n and page <= 10:
            batch = payload_rows(self.get("/tennis/v2/atp/ranking/singles", {"pageSize": 100, "pageNo": page}))
            if not batch:
                break
            rows.extend(batch)
            page += 1
        out = []
        for item in rows:
            p = item.get("player") if isinstance(item.get("player"), dict) else item
            name = _name(p) or _name(item)
            rank = _num(item.get("position", p.get("currentRank") if isinstance(p, dict) else None))
            pts = _num(item.get("rankingPoints", item.get("point", item.get("points"))))
            if name and rank:
                out.append({"player": name, "rank": int(rank), "points": pts or 0.0,
                            "matchstat_id": str((p or {}).get("id", item.get("playerId", "")))})
        return pd.DataFrame(out).drop_duplicates("player")

    # ------------------------------------------------------------- fixtures
    def upcoming(self, max_events: int = 300) -> list[dict]:
        """Upcoming ATP singles events normalised to {id, player_a, player_b, tournament, start_ts, round, status}."""
        raw: list[dict] = []
        for path, params in (
            ("/tennis/v2/extend/api/events/upcoming/atp", {"limit": 50}),
            ("/tennis/v2/upcoming/matches/atp", {"group": "singles", "limit": 50}),
        ):
            try:
                page = 1
                while len(raw) < max_events and page <= 8:
                    payload = self.get(path, {**params, "page": page})
                    batch = payload_rows(payload)
                    if not batch:
                        break
                    raw.extend(batch)
                    pag = payload.get("pagination", {}) if isinstance(payload, dict) else {}
                    if not pag.get("hasNext") and len(batch) < params["limit"]:
                        break
                    page += 1
                if raw:
                    break
            except Exception:
                raw = []
        return [e for e in (normalize_event(r) for r in raw) if e]

    def recent_results(self, days: int = 10) -> pd.DataFrame:
        """Completed ATP matches in the last ``days`` days, in the archive schema."""
        end = date.today()
        start = end - timedelta(days=days)
        records: list[dict] = []
        try:
            payload = self.first_working([
                f"/tennis/v2/atp/fixtures/{start.isoformat()}/{end.isoformat()}",
                f"/tennis/v2/atp/fixtures/{start.isoformat()}",
            ], {"include": "round,tournament,tournament.court,stat", "pageSize": 500})
            records = payload_rows(payload)
        except Exception as exc:
            print(f"WARNING fixtures-by-date unavailable ({exc}); falling back to player histories")
        if not records:
            records = self._results_from_player_histories(start)
        return results_to_frame(records, start)

    def _results_from_player_histories(self, since: date, top_n: int = 200) -> list[dict]:
        try:
            ranks = self.rankings(top_n)
        except Exception:
            return []
        seen, out = set(), []
        for mid in ranks.get("matchstat_id", []):
            if not mid:
                continue
            try:
                payload = self.first_working([
                    f"/tennis/v2/atp/player/past-matches/{mid}",
                    f"/tennis/v2/ms-api/atp/player/past-matches/{mid}",
                ], {"include": "round,tournament,tournament.court,tournament.rank,stat",
                    "filter": f"GameYear:{since.year}", "pageSize": 30, "pageNo": 1})
            except Exception:
                continue
            for item in payload_rows(payload):
                key = str(item.get("id") or (item.get("date"), item.get("player1Id"), item.get("player2Id")))
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return out

    # ----------------------------------------------------------------- odds
    def pre_match_odds(self, event_id) -> Any:
        return self.get(f"/tennis/v2/extend/api/odds/pre-match/{event_id}")


def normalize_event(row: dict) -> dict | None:
    if not isinstance(row, dict):
        return None
    a = _name(row.get("participant1") or row.get("player1") or row.get("homePlayer") or row.get("home"))
    b = _name(row.get("participant2") or row.get("player2") or row.get("awayPlayer") or row.get("away"))
    tour = row.get("tournament") or row.get("league") or row.get("competition")
    tournament = _name(tour) if not isinstance(tour, str) else tour
    if not (a and b and tournament):
        return None
    if "/" in a or "/" in b:  # doubles
        return None
    status = row.get("status") or row.get("matchStatus") or ""
    if isinstance(status, dict):
        status = status.get("name") or status.get("type") or ""
    rnd = row.get("round") or row.get("roundName") or ""
    if isinstance(rnd, dict):
        rnd = rnd.get("name") or ""
    surface = ""
    if isinstance(tour, dict):
        court = tour.get("court") if isinstance(tour.get("court"), dict) else {}
        surface = str(court.get("name") or tour.get("surface") or "")
    return {
        "id": str(row.get("id") or row.get("eventId") or row.get("matchId") or f"{a}|{b}"),
        "player_a": a, "player_b": b, "tournament": tournament, "surface_hint": surface,
        "start_ts": _ts(row.get("startTimestamp") or row.get("timestamp") or row.get("startTime") or row.get("date")),
        "round": str(rnd), "status": str(status or "Not started"), "source": "matchstat",
    }


def _surface(t: dict) -> tuple[str | None, str]:
    court = t.get("court") if isinstance(t.get("court"), dict) else {}
    name = str(court.get("name") or t.get("surface") or "").lower().replace(" ", "")
    indoor = "I" if name.startswith("i.") or "indoor" in name else "O"
    for s in ("hard", "clay", "grass"):
        if s in name:
            return s.title(), indoor
    if "carpet" in name:
        return "Hard", "I"
    return None, indoor


def results_to_frame(records: list[dict], since: date | None = None) -> pd.DataFrame:
    """Matchstat completed games -> archive schema (player1 = winner per Matchstat docs)."""
    rows = []
    for it in records:
        p1, p2, t = it.get("player1") or {}, it.get("player2") or {}, it.get("tournament") or {}
        if not all(isinstance(x, dict) for x in (p1, p2, t)):
            continue
        w, l = _name(p1), _name(p2)
        when = pd.to_datetime(it.get("date"), errors="coerce", utc=True)
        if not w or not l or pd.isna(when) or (since and when.date() < since):
            continue
        score = it.get("result") or it.get("score")
        if not score:
            continue
        surface, indoor = _surface(t)
        if surface is None:
            continue
        ws = p1.get("stats") if isinstance(p1.get("stats"), dict) else {}
        ls = p2.get("stats") if isinstance(p2.get("stats"), dict) else {}
        rnd = it.get("round") if isinstance(it.get("round"), dict) else {}

        def s(d, k):
            return _num(d.get(k)) if isinstance(d, dict) else None

        wbp, wbpo, lbp, lbpo = s(ws, "breakPointsConverted"), s(ws, "breakPointsConvertedOf"), s(ls, "breakPointsConverted"), s(ls, "breakPointsConvertedOf")
        rows.append({
            "tourney_id": str(t.get("id") or ""), "tourney_name": t.get("name"), "surface": surface,
            "tourney_level": "A", "indoor": indoor, "tourney_date": int(when.strftime("%Y%m%d")),
            "match_num": _num(it.get("id")), "winner_id": f"ms:{it.get('player1Id', p1.get('id'))}", "winner_name": w,
            "loser_id": f"ms:{it.get('player2Id', p2.get('id'))}", "loser_name": l, "score": score,
            "best_of": np.nan, "round": rnd.get("name") or it.get("roundName"),
            "w_ace": s(ws, "aces"), "w_df": s(ws, "doubleFaults"), "w_svpt": s(ws, "firstServeOf"), "w_1stIn": s(ws, "firstServe"),
            "w_1stWon": s(ws, "winningOnFirstServe"), "w_2ndWon": s(ws, "winningOnSecondServe"),
            "w_bpFaced": lbpo, "w_bpSaved": (lbpo - lbp) if lbpo is not None and lbp is not None else np.nan,
            "l_ace": s(ls, "aces"), "l_df": s(ls, "doubleFaults"), "l_svpt": s(ls, "firstServeOf"), "l_1stIn": s(ls, "firstServe"),
            "l_1stWon": s(ls, "winningOnFirstServe"), "l_2ndWon": s(ls, "winningOnSecondServe"),
            "l_bpFaced": wbpo, "l_bpSaved": (wbpo - wbp) if wbpo is not None and wbp is not None else np.nan,
            "w_winners": s(ws, "winners"), "w_unforced_errors": s(ws, "unforcedErrors"),
            "l_winners": s(ls, "winners"), "l_unforced_errors": s(ls, "unforcedErrors"),
            "data_source": "Matchstat",
        })
    return pd.DataFrame(rows)


def map_ids_to_archive(frame: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Give Matchstat players the archive's player ids (by name) so ratings continue."""
    if frame.empty:
        return frame
    ids = {}
    for side in ("winner", "loser"):
        part = archive[[f"{side}_name", f"{side}_id"]].dropna()
        ids.update(dict(zip(part[f"{side}_name"].map(normalize), part[f"{side}_id"].astype(str))))
    out = frame.copy()
    for side in ("winner", "loser"):
        mapped = out[f"{side}_name"].map(normalize).map(ids)
        out[f"{side}_id"] = mapped.fillna(out[f"{side}_id"])
    return out


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
