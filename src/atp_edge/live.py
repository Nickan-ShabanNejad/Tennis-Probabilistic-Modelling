"""Collect live fixtures and prices from the configured sources (or synthetic demo data)."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from .config import PREDICTIONS, secret
from .tournaments import resolve


CACHE_PATH = PREDICTIONS / "price_cache.json"


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def refresh_minutes(hours_to_start: float | None) -> float:
    """How stale a match's prices may get before they are pulled again.

    Prices move most in the last hours, so matches close to the start are refreshed
    every PRICE_INTERVAL_MINUTES (default 10), later ones less often.
    """
    fast = max(5.0, _env_num("PRICE_INTERVAL_MINUTES", 10))
    if hours_to_start is None:
        return max(fast, 60)
    if hours_to_start <= 3:
        return fast
    if hours_to_start <= 12:
        return max(fast, 30)
    return max(fast, 120)


def is_due(fetched: float | None, start_ts: float | None, now: float, slow: float = 1.0) -> bool:
    if not fetched:
        return True
    hours = (start_ts - now) / 3600 if start_ts else None
    return now - fetched >= refresh_minutes(hours) * 60 * slow - 90   # 90 s slack for tick jitter


def load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, separators=(",", ":"), default=str))


def matchstat_budget(quota: dict | None, now: float) -> int:
    """Odds calls allowed this run, from the RapidAPI plan usage headers.

    Spreads what is left of the plan evenly until it resets, keeping a reserve for the
    daily results job. Without header information: MATCHSTAT_MAX_CALLS_PER_RUN (40).
    """
    cap = int(_env_num("MATCHSTAT_MAX_CALLS_PER_RUN", 40))
    q = quota or {}
    remaining, limit, reset_s = q.get("remaining"), q.get("limit"), q.get("reset_s")
    if remaining is None:
        return cap
    reserve = max(300, int(0.05 * (limit or 0)))
    spare = remaining - reserve
    if spare <= 0:
        return 0
    if not reset_s:
        return min(cap, spare)
    left = max(60.0, reset_s - (now - float(q.get("at") or now)))
    runs_left = max(1.0, left / (60 * max(5.0, _env_num("PRICE_INTERVAL_MINUTES", 10))))
    return int(max(1, min(cap, spare / runs_left)))


def collect(rt, demo: bool = False, persist: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Fixtures + prices for the next 48 hours, refreshing each source only when due.

    A price cache (data/predictions/price_cache.json) remembers when every tournament
    (The Odds API) and every match (Matchstat) was last priced, so a run every 10 minutes
    pulls fresh prices for matches about to start without spending quota on matches
    two days away.
    """
    if demo:
        return demo_feed(rt)
    now = time.time()
    cache = load_cache()
    status: dict = {}

    # ---------------------------------------------------------------- fixtures
    fixtures: list[dict] = (cache.get("fixtures") or {}).get("items", [])
    ms = None
    if secret("RAPIDAPI_KEY"):
        try:
            from .rapidapi import MatchstatClient
            ms = MatchstatClient()
            ms.quota = dict(cache.get("matchstat_quota") or {})
            fx_age = now - float((cache.get("fixtures") or {}).get("fetched") or 0)
            if fx_age >= 60 * _env_num("FIXTURE_REFRESH_MINUTES", 30) - 90 or not fixtures:
                fresh = ms.upcoming()
                if fresh:
                    fixtures = [f for f in fresh if resolve(f.get("tournament", "")) is not None]
                    cache["fixtures"] = {"fetched": now, "items": fixtures}
                status["matchstat"] = {"ok": True, "fixtures": len(fixtures), "refreshed": True, "api_calls": ms.calls}
            else:
                status["matchstat"] = {"ok": True, "fixtures": len(fixtures), "refreshed": False,
                                       "age_minutes": round(fx_age / 60)}
        except Exception as exc:
            status["matchstat"] = {"ok": False, "error": str(exc)[:300]}
    else:
        status["matchstat"] = {"ok": False, "error": "RAPIDAPI_KEY not set"}

    # ----------------------------------------------------- The Odds API (per tournament)
    odds_cache: dict = cache.get("odds_api") or {}
    if secret("ODDS_API_KEY"):
        try:
            from .odds_api import OddsAPI, parse_event
            api = OddsAPI()
            sports = [s for s in api.atp_sports() if s["event"] is not None]   # free call
            remaining = float(api.remaining) if api.remaining not in (None, "") else None
            reserve = _env_num("ODDS_API_RESERVE", 1500)
            slow = 1.0 if remaining is None or remaining > reserve else (3.0 if remaining > 300 else None)
            pulled, credits_before = [], api.used
            for s in sports:
                ent = odds_cache.get(s["key"]) or {}
                starts = [e.get("start_ts") for e in ent.get("events", []) if e.get("start_ts") and e["start_ts"] > now - 600]
                if slow is None or not is_due(ent.get("fetched"), min(starts) if starts else None, now, slow):
                    continue
                events = [parse_event(e, s["title"]) for e in api.odds(s["key"], slam=(s["event"].level == "G"))]
                odds_cache[s["key"]] = {"fetched": now, "title": s["title"], "events": events}
                pulled.append(s["title"])
            active = {s["key"] for s in sports}
            odds_cache = {k: v for k, v in odds_cache.items() if k in active}
            cache["odds_api"] = odds_cache
            status["odds_api"] = {"ok": True, "tournaments": [s["title"] for s in sports], "refreshed": pulled,
                                  "credits_remaining": api.remaining, "credits_used": api.used,
                                  "note": None if slow == 1.0 else ("credits low: refreshing 3x less often" if slow else
                                                                   "credits nearly exhausted: price pulls paused")}
        except Exception as exc:
            status["odds_api"] = {"ok": False, "error": str(exc)[:300]}
    else:
        status["odds_api"] = {"ok": False, "error": "ODDS_API_KEY not set"}
    odds_events: list[dict] = []
    for ent in odds_cache.values():
        for e in ent.get("events", []):
            if e.get("start_ts") and e["start_ts"] < now - 600:
                continue
            odds_events.append({**e, "prices_ts": ent.get("fetched")})

    # ------------------------------------------------ Matchstat odds (per match, fallback)
    if ms is not None:
        status["matchstat_odds"] = add_matchstat_prices(rt, ms, fixtures, odds_events, cache, now)
        cache["matchstat_quota"] = ms.quota
        status["matchstat_quota"] = ms.quota
    if persist:
        save_cache(cache)
    return fixtures, odds_events, status


def add_matchstat_prices(rt, ms, fixtures: list[dict], odds_events: list[dict], cache: dict, now: float,
                         horizon_hours: float = 48) -> dict:
    """Prices from Matchstat for tour-level fixtures The Odds API does not cover.

    Matches are refreshed soonest-first, each only when its prices are due, within a
    per-run budget that keeps the RapidAPI plan from running out.
    """
    from .matchstat_odds import fetch_event_odds
    from .predict import pair_key
    covered = {pair_key(o["player_a"], o["player_b"]) for o in odds_events if o.get("prices_a")}
    mcache: dict = cache.get("matchstat_odds") or {}
    budget = matchstat_budget(ms.quota, now)
    start_calls = ms.calls
    candidates = []
    for f in fixtures:
        ev = resolve(f.get("tournament", ""))
        ts = f.get("start_ts") or 0
        if ev is None or pair_key(f["player_a"], f["player_b"]) in covered:
            continue
        if ts and not (now - 600 < ts < now + horizon_hours * 3600):
            continue
        if rt.engine.find(f["player_a"]) is None or rt.engine.find(f["player_b"]) is None:
            continue
        candidates.append((ts or now + horizon_hours * 3600, f, ev))
    candidates.sort(key=lambda c: c[0])
    refreshed = priced = skipped_budget = 0
    errors: list[str] = []
    for ts, f, ev in candidates:
        key = str(f["id"])
        ent = mcache.get(key) or {}
        if is_due(ent.get("fetched"), ts, now):
            if ms.calls - start_calls >= budget:
                skipped_budget += 1
            else:
                got = fetch_event_odds(ms, f["id"], want_sets=(ev.level == "G"))
                refreshed += 1
                if got["prices_a"]:
                    ent = {"fetched": now, "start_ts": ts, "prices_a": got["prices_a"], "prices_b": got["prices_b"],
                           "sets35": got.get("sets35") or {}}
                else:
                    ent = {**ent, "fetched": now, "start_ts": ts}   # keep the last prices we had
                    errors.extend((got.get("errors") or [])[:1])
                mcache[key] = ent
        if ent.get("prices_a"):
            f["prices_a"], f["prices_b"] = ent["prices_a"], ent["prices_b"]
            if ent.get("sets35"):
                f["sets35"] = ent["sets35"]
            f["odds_source"], f["prices_ts"] = "matchstat", ent.get("fetched")
            priced += 1
    cache["matchstat_odds"] = {k: v for k, v in mcache.items() if (v.get("start_ts") or now) > now - 6 * 3600}
    return {"ok": True, "fixtures_with_prices": priced, "refreshed": refreshed, "waiting_for_budget": skipped_budget,
            "budget_this_run": budget, "api_calls": ms.calls - start_calls, "sample_errors": errors[:3]}


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def demo_feed(rt) -> tuple[list[dict], list[dict], dict]:
    """Synthetic slate: a Masters event and a Grand Slam, prices = ratings + noise + margin."""
    eng = rt.engine
    recent = [p for p in eng.players.values() if p.last_date and eng.last_date and (eng.last_date - p.last_date).days < 60]
    recent.sort(key=lambda p: -p.elo)
    pool = recent[:32]
    rng = np.random.default_rng(_seed(time.strftime("%Y-%m-%d")))
    rng.shuffle(pool)
    now = time.time()
    base = now - (now % 3600) + 2 * 3600
    odds_events = []
    for i in range(0, min(len(pool), 24) - 1, 2):
        a, b = pool[i], pool[i + 1]
        tournament = "Australian Open" if i < 12 else "Paris Masters"
        ev = resolve(tournament)
        se = 0.5 * (a.elo + a.surf_elo.get(ev.surface, 1500)) - 0.5 * (b.elo + b.surf_elo.get(ev.surface, 1500))
        p = 1 / (1 + 10 ** (-se / 400))
        if ev.best_of == 5:
            from .betting import bo5_win, set_prob_from_match
            p = bo5_win(set_prob_from_match(p, 3))
        r = np.random.default_rng(_seed(a.name + b.name))
        p = float(np.clip(p + r.normal(0, 0.035), 0.04, 0.96))

        def price(q, margin):
            return round(1 / (q * margin), 2)

        books = {"pinnacle": 1.025, "betway": 1.06, "unibet": 1.055, "williamhill": 1.065}
        pa = {k: price(p, m * float(r.uniform(0.97, 1.03))) for k, m in books.items()}
        pb = {k: price(1 - p, m * float(r.uniform(0.97, 1.03))) for k, m in books.items()}
        sets35 = {"over": {}, "under": {}}
        if ev.level == "G":
            from .betting import over35_from_set_prob, set_prob_from_match
            iid = over35_from_set_prob(set_prob_from_match(max(p, 1 - p), 5))
            q = float(np.clip(0.5 * iid + 0.5 * 0.5 - 0.04 + r.normal(0, 0.04), 0.2, 0.8))
            sets35 = {"over": {"pinnacle": price(q, 1.03), "unibet": price(q, 1.06)},
                      "under": {"pinnacle": price(1 - q, 1.03), "unibet": price(1 - q, 1.06)}}
        odds_events.append({"odds_event_id": f"demo-{i}", "player_a": a.name, "player_b": b.name,
                            "start_ts": base + (i // 2) * 3600, "tournament": tournament, "round": "R32",
                            "prices_a": pa, "prices_b": pb, "sets35": sets35, "source": "demo"})
    return [], odds_events, {"demo": {"ok": True, "events": len(odds_events), "note": "SYNTHETIC prices"}}
