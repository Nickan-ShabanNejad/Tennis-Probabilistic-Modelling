"""The Odds API (https://the-odds-api.com) client - live prices.

Key: ODDS_API_KEY. Quota: each request costs (#markets x #regions) credits. With
the defaults (region "eu" which contains Pinnacle, moneyline only, sets totals
added for Grand Slams) a run costs roughly 1-4 credits; a run every 4 hours fits
the free 500-credit monthly plan. Set ODDS_API_REGIONS="eu,us" to also shop North
American books (doubles the cost).

Which tennis tournaments the API carries is decided by the provider; usually
Slams, Masters and many 500/250 events. Matches without a price still get a
model probability on the board.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from .config import secret
from .tournaments import resolve

BASE = "https://api.the-odds-api.com/v4"
SHARP_BOOK = "pinnacle"
EXCHANGES = {"betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au", "matchbook", "smarkets"}


class OddsAPI:
    def __init__(self, api_key: str | None = None, regions: str | None = None, timeout: float = 30.0):
        self.key = (api_key or secret("ODDS_API_KEY")).strip()
        if not self.key:
            raise ValueError("ODDS_API_KEY is not set")
        self.regions = regions or os.getenv("ODDS_API_REGIONS", "eu")
        self.timeout = timeout
        self.remaining: str | None = None
        self.used: str | None = None

    def _get(self, path: str, **params) -> Any:
        r = requests.get(f"{BASE}/{path.lstrip('/')}", params={"apiKey": self.key, **params}, timeout=self.timeout)
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        self.used = r.headers.get("x-requests-used", self.used)
        r.raise_for_status()
        return r.json()

    def atp_sports(self) -> list[dict]:
        """Active men's singles tournaments (free call - no quota cost)."""
        sports = self._get("sports")
        out = []
        for s in sports:
            key, title = str(s.get("key", "")), str(s.get("title", ""))
            if not key.startswith("tennis_atp") or not s.get("active") or s.get("has_outrights"):
                continue
            if "doubles" in key or "doubles" in title.lower():
                continue
            ev = resolve(title) or resolve(key.replace("tennis_atp_", "").replace("_", " "))
            out.append({"key": key, "title": title, "event": ev})
        return out

    def odds(self, sport_key: str, slam: bool = False) -> list[dict]:
        markets = "h2h,totals" if slam else "h2h"
        return self._get(f"sports/{sport_key}/odds", regions=self.regions, markets=markets,
                         oddsFormat="decimal", dateFormat="unix")


def parse_event(ev: dict, sport_title: str = "") -> dict:
    """One Odds API event -> prices per player and Grand Slam set totals."""
    home, away = ev.get("home_team", ""), ev.get("away_team", "")
    prices: dict[str, dict[str, float]] = {home: {}, away: {}}
    sets35: dict[str, dict[str, float]] = {"over": {}, "under": {}}
    for book in ev.get("bookmakers", []) or []:
        bkey = book.get("key", "")
        for market in book.get("markets", []) or []:
            mkey = market.get("key")
            for o in market.get("outcomes", []) or []:
                price = o.get("price")
                if not price:
                    continue
                if mkey == "h2h" and o.get("name") in prices:
                    prices[o["name"]][bkey] = float(price)
                elif mkey == "totals":
                    # In a best-of-5 match a 3.5 line can only be total SETS (games lines are ~38.5).
                    try:
                        point = float(o.get("point"))
                    except Exception:
                        continue
                    if abs(point - 3.5) < 1e-9 and str(o.get("name", "")).lower() in ("over", "under"):
                        sets35[str(o["name"]).lower()][bkey] = float(price)
    return {
        "odds_event_id": ev.get("id"), "player_a": home, "player_b": away, "start_ts": ev.get("commence_time"),
        "tournament": sport_title or ev.get("sport_title", ""), "prices_a": prices[home], "prices_b": prices[away],
        "sets35": sets35, "source": "the-odds-api",
    }


def best_price(prices: dict[str, float], exclude_exchanges: bool = True) -> tuple[float | None, str | None]:
    items = [(v, k) for k, v in prices.items() if not (exclude_exchanges and k in EXCHANGES)]
    if not items:
        return None, None
    v, k = max(items)
    return v, k
