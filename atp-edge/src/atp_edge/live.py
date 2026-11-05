"""Collect live fixtures and prices from the configured sources (or synthetic demo data)."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .config import secret
from .tournaments import resolve


def collect(rt, demo: bool = False) -> tuple[list[dict], list[dict], dict]:
    if demo:
        return demo_feed(rt)
    status: dict = {}
    fixtures: list[dict] = []
    odds_events: list[dict] = []
    if secret("RAPIDAPI_KEY"):
        try:
            from .rapidapi import MatchstatClient
            ms = MatchstatClient()
            fixtures = ms.upcoming()
            status["matchstat"] = {"ok": True, "fixtures": len(fixtures), "api_calls": ms.calls}
        except Exception as exc:
            status["matchstat"] = {"ok": False, "error": str(exc)}
    else:
        status["matchstat"] = {"ok": False, "error": "RAPIDAPI_KEY not set"}
    if secret("ODDS_API_KEY"):
        try:
            from .odds_api import OddsAPI, parse_event
            api = OddsAPI()
            sports = api.atp_sports()
            for s in sports:
                ev = s["event"]
                if ev is None:
                    continue
                for e in api.odds(s["key"], slam=(ev.level == "G")):
                    odds_events.append(parse_event(e, s["title"]))
            status["odds_api"] = {"ok": True, "tournaments": [s["title"] for s in sports], "events": len(odds_events),
                                  "credits_remaining": api.remaining, "credits_used": api.used}
        except Exception as exc:
            status["odds_api"] = {"ok": False, "error": str(exc)}
    else:
        status["odds_api"] = {"ok": False, "error": "ODDS_API_KEY not set"}
    return fixtures, odds_events, status


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
