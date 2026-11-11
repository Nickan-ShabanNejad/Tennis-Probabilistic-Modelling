"""Written match previews.

* ``template_preview`` - always available, built from the model's own numbers.
* ``ai_previews`` - optional. With an ANTHROPIC_API_KEY secret, the live job asks
  Claude (Haiku) to write a short preview from the same facts. Previews are cached
  in data/predictions/previews.json and only regenerated when the numbers move, so
  the cost is a fraction of a cent per match.

The AI only rewrites the facts it is given; it is told not to invent injuries,
results or news it was not given.
"""
from __future__ import annotations

import json
import os
import time

import requests

from .config import PREDICTIONS, secret

CACHE = PREDICTIONS / "previews.json"
MODEL = os.getenv("PREVIEW_MODEL", "claude-haiku-4-5-20251001")


def _pct(x) -> str:
    return f"{float(x):.0%}" if x is not None else "n/a"


def facts(m: dict) -> dict:
    a, b = m["player_a"], m["player_b"]
    sa, sb = m.get("stats_a") or {}, m.get("stats_b") or {}
    rec = m.get("recommendation")
    sets = m.get("sets") or {}
    top = list((m.get("factors") or {}).items())[:4]
    return {
        "match": f"{a} vs {b}", "event": m.get("event"), "surface": m.get("surface"), "indoor": m.get("indoor"),
        "best_of": m.get("best_of"), "court_speed_index": round(float(m.get("court_speed") or 1), 2),
        "model_win_probability": {a: round(m["p_model_a"], 3), b: round(1 - m["p_model_a"], 3)},
        "market_win_probability": None if m.get("p_market_a") is None else {a: round(m["p_market_a"], 3), b: round(1 - m["p_market_a"], 3)},
        "fair_win_probability_used_for_bets": {a: round(m.get("p_fair_a", m["p_model_a"]), 3), b: round(1 - m.get("p_fair_a", m["p_model_a"]), 3)},
        "biggest_model_factors_log_odds_positive_favours_" + a: {k: round(v, 3) for k, v in top},
        "elo": {a: round(sa.get("Elo (overall)") or 0), b: round(sb.get("Elo (overall)") or 0)},
        "rank": {a: m.get("rank_a"), b: m.get("rank_b")},
        "head_to_head": {"overall": f"{m['h2h'][0]}-{m['h2h'][1]}", "this_surface": f"{m['h2h_surface'][0]}-{m['h2h_surface'][1]}"},
        "history_at_event": {a: m.get("event_record_a"), b: m.get("event_record_b")},
        "injury_signals": {a: m.get("injury_signals_a") or "none", b: m.get("injury_signals_b") or "none"},
        "recommended_bet": None if not rec else f"{rec['selection']} @ {rec['price']:.2f}, EV {rec['ev']:+.1%}",
        "over_3_5_sets_probability": sets.get("p_over35"),
    }


def template_preview(m: dict) -> str:
    a, b = m["player_a"], m["player_b"]
    pa = float(m["p_model_a"])
    fav, dog, pf = (a, b, pa) if pa >= 0.5 else (b, a, 1 - pa)
    parts = [f"The model makes {fav} a {pf:.0%} favourite over {dog}"]
    if m.get("p_market_a") is not None:
        pm = float(m["p_market_a"]) if fav == a else 1 - float(m["p_market_a"])
        gap = pf - pm
        parts[0] += (f", {abs(gap):.0%} {'more' if gap > 0 else 'less'} confident than the market ({pm:.0%})" if abs(gap) >= 0.02
                     else f", in line with the market ({pm:.0%})")
    text = parts[0] + "."
    f = [(k, v) for k, v in (m.get("factors") or {}).items() if abs(v) >= 0.05][:3]
    if f:
        bits = [f"{k.lower()} ({'favours ' + (a if v > 0 else b)})" for k, v in f]
        text += " The main drivers are " + ", ".join(bits) + "."
    sp = float(m.get("court_speed") or 1)
    text += (f" Conditions: {m.get('event')} plays {'fast' if sp >= 1.15 else 'slow' if sp < 0.9 else 'medium'} "
             f"(court speed {sp:.2f}){', indoors' if m.get('indoor') else ''}.")
    h = m.get("h2h") or (0, 0)
    if sum(h):
        text += f" They are {h[0]}-{h[1]} head to head."
    inj = [f"{n}: {s}" for n, s in ((a, m.get("injury_signals_a")), (b, m.get("injury_signals_b"))) if s]
    if inj:
        text += " Injury watch – " + "; ".join(inj) + "."
    sets = m.get("sets") or {}
    if sets.get("available"):
        text += f" Chance of four or more sets: {sets['p_over35']:.0%}."
    rec = m.get("recommendation")
    text += (f" Signal: back {rec['selection']} at {rec['price']:.2f} ({rec['ev']:+.1%} EV)." if rec else " No bet signal at current prices.")
    return text


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def ai_previews(matches: list[dict], max_new: int = 30) -> dict:
    """Generate/refresh cached AI previews. Returns {match_id: text}."""
    from .tracking import match_id

    key = secret("ANTHROPIC_API_KEY")
    cache = _load()
    status = {"enabled": bool(key), "new": 0, "cached": 0, "errors": 0}
    if not key:
        return status
    fresh = {}
    for m in matches:
        mid = match_id(m)
        # Coarse signature: prices now refresh every 10 minutes, and a preview is only rewritten
        # when the picture really changes (5-point moves or a bet appearing/disappearing).
        pm = m.get("p_market_a")
        sig = f"{round(m['p_model_a'] * 20)}|{None if pm is None else round(pm * 20)}|{bool(m.get('recommendation'))}"
        old = cache.get(mid)
        if old and old.get("sig") == sig:
            fresh[mid] = old
            status["cached"] += 1
            continue
        if status["new"] >= max_new:
            if old:
                fresh[mid] = old
            continue
        prompt = (
            "Write a 3-4 sentence preview of this ATP tennis match for a bettor, in plain English. Use ONLY the facts "
            "below; do not invent injuries, results, news or statistics. Mention who the model favours and why, the "
            "conditions, and whether there is a bet signal. No headings, no bullet points.\n\n"
            + json.dumps(facts(m), default=str))
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": MODEL, "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = "".join(c.get("text", "") for c in r.json().get("content", []) if c.get("type") == "text").strip()
            if text:
                fresh[mid] = {"sig": sig, "text": text, "at": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
                status["new"] += 1
        except Exception as exc:
            status["errors"] += 1
            status["last_error"] = str(exc)[:200]
            if old:
                fresh[mid] = old
    CACHE.write_text(json.dumps(fresh, indent=1))
    return status


def cached_preview(m: dict) -> str | None:
    from .tracking import match_id
    return (_load().get(match_id(m)) or {}).get("text")
