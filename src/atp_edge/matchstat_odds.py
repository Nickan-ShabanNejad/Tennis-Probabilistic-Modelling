"""Bookmaker prices from Matchstat (RapidAPI) - fallback when The Odds API does not
carry a tournament (it often skips ATP 250/500 events).

Matchstat odds payloads come in several shapes:
  {"result": {"Pinnacle": {"Full Time Result": [{"od1":..,"od2":..,"sourceAddTime":..}, ...]}}}
  {"result": {"Bet365": {"Full Time Result": {"start": {...}, "latest": {...}}}}}
  [{"bookmaker": "Pinnacle", "market": "Match Winner", "od1":.., "od2":..}, ...]
For a two-way match-winner market od1 = participant 1 = the fixture's player_a.
"""
from __future__ import annotations

import re
from typing import Any

MONEYLINE_MARKETS = ("full time result", "match winner", "moneyline", "home/away", "winner", "to win match", "1x2")


def _dec(v) -> float | None:
    try:
        f = float(v)
        return f if 1.0 < f < 1000 else None
    except Exception:
        return None


def book_key(name: str) -> str:
    """'Pinnacle' -> 'pinnacle', 'William Hill' -> 'williamhill' (Odds-API style keys)."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _pair(row: dict) -> tuple[float, float] | None:
    for k1, k2 in (("od1", "od2"), ("odds1", "odds2"), ("currentOd1", "currentOd2"), ("latestOd1", "latestOd2"),
                   ("homeOdds", "awayOdds"), ("outcome1Odds", "outcome2Odds")):
        a, b = _dec(row.get(k1)), _dec(row.get(k2))
        if a and b:
            return a, b
    for key in ("outcomes", "selections", "prices"):
        items = row.get(key)
        if isinstance(items, list):
            vals = [_dec(i.get("odds") or i.get("price") or i.get("value")) for i in items if isinstance(i, dict)]
            vals = [v for v in vals if v]
            if len(vals) == 2:
                return vals[0], vals[1]
    return None


def _latest(quotes: Any) -> tuple[float, float] | None:
    """Most recent quote from a list (by sourceAddTime) or a {start/latest/current} dict."""
    if isinstance(quotes, list):
        rows = [q for q in quotes if isinstance(q, dict) and _pair(q)]
        if not rows:
            return None
        rows.sort(key=lambda q: float(q.get("sourceAddTime") or q.get("time") or q.get("timestamp") or 0))
        return _pair(rows[-1])
    if isinstance(quotes, dict):
        direct = _pair(quotes)
        if direct:
            return direct
        for k in ("latest", "current", "last", "end", "start", "opening"):
            if isinstance(quotes.get(k), dict) and _pair(quotes[k]):
                return _pair(quotes[k])
    return None


def _is_moneyline(name: str) -> bool:
    n = str(name or "").lower()
    return any(m in n for m in MONEYLINE_MARKETS) and "set" not in n and "game" not in n and "handicap" not in n


def _ts(row: dict) -> float:
    for k in ("addTime", "sourceAddTime", "time", "timestamp", "updatedAt"):
        try:
            return float(row.get(k))
        except Exception:
            continue
    return 0.0


def _rows_by_book(rows: list, line: float | None = None) -> dict[str, tuple[float, float]]:
    """Latest two-way quote per bookmaker from market rows like
    {"bookmaker": "Pinnacle", "od1": "1.9", "od2": "1.95", "addTime": ..., "line": "2.5"}."""
    best: dict[str, tuple[float, tuple[float, float]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("bookmaker") or str(row.get("oddType", "pre")).lower() not in ("pre", ""):
            continue
        if line is not None:
            try:
                if abs(float(row.get("line")) - line) > 1e-9:
                    continue
            except Exception:
                continue
        pr = _pair(row)
        if not pr:
            continue
        key, ts = book_key(row["bookmaker"]), _ts(row)
        if key not in best or ts >= best[key][0]:
            best[key] = (ts, pr)
    return {k: v[1] for k, v in best.items()}


def parse_moneylines(payload: Any) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    root = payload.get("result", payload.get("data", payload)) if isinstance(payload, dict) else payload
    # Shape seen live (Nov 2026): {"result": {"Full Time Result": [{"bookmaker": ..., "od1": ..}, ..], ...}}
    if isinstance(root, dict):
        for market, rows in root.items():
            if isinstance(rows, list) and _is_moneyline(market):
                out.update(_rows_by_book(rows))
        if out:
            return out
    if isinstance(root, dict):
        for book, markets in root.items():
            if not isinstance(markets, dict):
                continue
            for market, quotes in markets.items():
                if _is_moneyline(market):
                    pr = _latest(quotes)
                    if pr:
                        out[book_key(book)] = pr
                        break
    rows = root if isinstance(root, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        book = row.get("bookmaker") or row.get("bookmakerName") or row.get("book")
        if isinstance(book, dict):
            book = book.get("name")
        market = str(row.get("market") or row.get("marketName") or "Match Winner")
        if book and _is_moneyline(market):
            pr = _pair(row) or _latest(row.get("odds"))
            if pr:
                out[book_key(book)] = pr
    return out


def parse_sets35(payload: Any) -> dict[str, dict[str, float]]:
    """Over/Under 3.5 sets by book (Grand Slam best-of-five)."""
    res = {"over": {}, "under": {}}
    root = payload.get("result", payload.get("data", payload)) if isinstance(payload, dict) else payload
    if not isinstance(root, dict):
        return res
    # Live shape: market "Over Under" rows with line "3.5" (a line this small can only be sets).
    for market, rows in root.items():
        m = str(market).lower()
        if isinstance(rows, list) and ("over" in m or "total" in m) and "game" not in m:
            for book, (o, u) in _rows_by_book(rows, line=3.5).items():
                res["over"][book], res["under"][book] = o, u
    if res["over"]:
        return res
    for book, markets in root.items():
        if not isinstance(markets, dict):
            continue
        for market, quotes in markets.items():
            m = str(market).lower()
            if "set" in m and "3.5" in m:
                pr = _latest(quotes)
                if pr:
                    first_is_under = "under" in m.split("3.5")[0][-12:] if "under" in m else False
                    over, under = (pr[1], pr[0]) if first_is_under else (pr[0], pr[1])
                    res["over"][book_key(book)] = over
                    res["under"][book_key(book)] = under
    return res


def fetch_event_odds(client, event_id, want_sets: bool = False) -> dict:
    """Try Matchstat's odds endpoints in turn; returns prices oriented to participant 1/2."""
    paths = (
        (f"/tennis/v2/extend/api/odds/pre-match/{event_id}", None),
        (f"/tennis/v2/extend/api/odds/compare/{event_id}", {"market_id": 1}),
        (f"/tennis/v2/extend/api/event/recent-odds/get/{event_id}", None),
    )
    errors = []
    for path, params in paths:
        try:
            payload = client.get(path, params)
        except Exception as exc:
            errors.append(str(exc)[:120])
            continue
        ml = parse_moneylines(payload)
        if not ml and payload:
            _save_sample(path, payload)
        if ml:
            return {"prices_a": {k: v[0] for k, v in ml.items()}, "prices_b": {k: v[1] for k, v in ml.items()},
                    "sets35": parse_sets35(payload) if want_sets else {}, "source": path.split("/api/")[-1].split("/")[0] + ":" + path.split("/")[-2]}
    return {"prices_a": {}, "prices_b": {}, "sets35": {}, "errors": errors}


def _save_sample(path: str, payload: Any) -> None:
    """Keep one unparsed payload so an unexpected response shape can be diagnosed from the repo."""
    import json
    from .config import GENERATED
    target = GENERATED / "matchstat_odds_sample.json"
    try:
        if not target.exists():
            target.write_text(json.dumps({"endpoint": path, "payload": payload}, default=str)[:60000])
    except Exception:
        pass
