"""Turn fixtures + prices into predictions and bet recommendations."""
from __future__ import annotations

import json
import math
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from . import injuries as injury_flags
from .betting import injury_adjust, kelly_stake, no_vig, set_prob_from_match, sets_distribution
from .config import APP_TIMEZONE, GENERATED
from .engine import ROUND_INDEX
from .modeling import ENGINE_PATH, MATCH_MODEL, METRICS_PATH, predict_over35
from .odds_api import SHARP_BOOK, best_price
from .tournaments import BETTABLE_LEVELS, normalize, resolve

FAMILY_OF = {
    # every feature belongs to exactly one family (no double counting)
    "elo_diff": "Ratings (Elo)", "surf_elo_diff": "Ratings (Elo)", "blend_elo_diff": "Ratings (Elo)",
    "elo_prob_logit": "Ratings (Elo)", "cond_elo_diff": "Ratings (Elo)", "exp_diff": "Ratings (Elo)", "surf_exp_diff": "Ratings (Elo)",
    "log_rank_diff": "Ratings (Elo)", "level_elo": "Ratings (Elo)", "bo5_elo": "Ratings (Elo)",
    "serve_diff": "Serve", "spw10_diff": "Serve", "spw_adj10_diff": "Serve", "ace10_diff": "Serve", "df10_diff": "Serve", "indoor_serve": "Serve",
    "return_diff": "Return", "rpw10_diff": "Return", "rpw_adj10_diff": "Return",
    "win10_diff": "Recent form", "surf_win10_diff": "Recent form", "form_diff": "Recent form",
    "opp10_diff": "Recent form", "elo_trend_diff": "Recent form",
    "m7_diff": "Fatigue & schedule", "m14_diff": "Fatigue & schedule", "sets7_diff": "Fatigue & schedule",
    "min7_diff": "Fatigue & schedule", "rest_diff": "Fatigue & schedule",
    "inj_recent_diff": "Injury signals", "inj_count_diff": "Injury signals", "layoff_diff": "Injury signals",
    "comeback_diff": "Injury signals",
    "event_resid_diff": "History at this event", "event_win_diff": "History at this event",
    "level_resid_diff": "Event type (level / best-of-5)", "bo5_resid_diff": "Event type (level / best-of-5)",
    "speed_fit_diff": "Court speed", "speed_serve": "Court speed", "speed_return": "Court speed",
    "speed_ace": "Court speed", "speed_elo": "Court speed",
    "h2h_edge": "Head to head", "h2h_surf_edge": "Head to head",
    "age_diff": "Age & physique", "height_diff": "Age & physique",
}




@dataclass
class Settings:
    """Betting rule. The automated bet log uses these defaults, which can be changed
    with repository variables (MIN_EV, MIN_EDGE, MIN_EV_SETS, KELLY_FRACTION,
    KELLY_CAP, BET_BOOKS) without touching code."""
    strategy: str = "model"           # the model's probability is the fair price
    min_ev: float = field(default_factory=lambda: _env_float("MIN_EV", 0.05))
    min_edge: float = field(default_factory=lambda: _env_float("MIN_EDGE", 0.05))
    min_ev_sets: float = field(default_factory=lambda: _env_float("MIN_EV_SETS", 0.05))
    kelly_fraction: float = field(default_factory=lambda: _env_float("KELLY_FRACTION", 0.25))
    kelly_cap: float = field(default_factory=lambda: _env_float("KELLY_CAP", 0.02))
    horizon_hours: float = 48.0
    books: tuple = field(default_factory=lambda: _env_books())   # empty = every book


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_books() -> tuple:
    return tuple(b.strip().lower() for b in os.getenv("BET_BOOKS", "").split(",") if b.strip())


def _allowed(prices: dict, books: tuple) -> dict:
    return {k: v for k, v in (prices or {}).items() if not books or k in books}


class Runtime:
    def __init__(self):
        self.engine = joblib.load(ENGINE_PATH)
        self.model = joblib.load(MATCH_MODEL)
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
        self.injuries = injury_flags.load()

    # ------------------------------------------------------------------
    def explain(self, x: dict) -> dict:
        pipe = self.model.pipeline
        steps = getattr(pipe, "named_steps", {})
        if "scale" not in steps or not hasattr(steps.get("model"), "coef_"):
            return {}
        names = self.model.features
        v = np.array([[x.get(f, np.nan) for f in names]], float)
        v = steps["impute"].transform(v)
        m = v.copy()
        m[:, self.model.diff_idx] *= -1
        z, zm = steps["scale"].transform(v)[0], steps["scale"].transform(m)[0]
        contrib = steps["model"].coef_[0] * (z - zm) / 2.0
        out: dict[str, float] = {}
        for i, f in enumerate(names):
            if f.startswith("ctx_"):
                continue
            fam = FAMILY_OF.get(f, "Other")
            out[fam] = out.get(fam, 0.0) + float(contrib[i])
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))

    # ------------------------------------------------------------------
    def predict(self, a_name: str, b_name: str, tournament: str, start_ts: float | None, round_name: str = "",
                prices_a: dict | None = None, prices_b: dict | None = None, sets35: dict | None = None,
                settings: Settings | None = None) -> dict:
        st = settings or Settings()
        ev = resolve(tournament)
        base = {"player_a": a_name, "player_b": b_name, "tournament": tournament, "start_ts": start_ts, "round": round_name}
        if ev is None:
            return {**base, "status": "skipped", "reason": "not an ATP 250/500/1000/Slam/Finals event"}
        pa, pb = self.engine.find(a_name), self.engine.find(b_name)
        if pa is None or pb is None:
            missing = [n for n, p in ((a_name, pa), (b_name, pb)) if p is None]
            return {**base, "status": "skipped", "reason": f"no match history for {', '.join(missing)}",
                    "event": ev.name, "level": ev.level, "surface": ev.surface}
        when = datetime.fromtimestamp(start_ts, tz=timezone.utc) if start_ts else datetime.now(timezone.utc)
        ctx = {"tournament": ev.key, "surface": ev.surface, "level": ev.level, "best_of": ev.best_of,
               "indoor": ev.indoor, "round": _round_code(round_name), "date": when.date()}
        out = self.engine.features(pa.pid, pb.pid, ctx)
        x, meta = out["x"], out["meta"]
        p_model = float(self.model.predict_rows([x])[0])
        sev_a = injury_flags.severity(pa.name, self.injuries)
        sev_b = injury_flags.severity(pb.name, self.injuries)
        p_model_adj = injury_adjust(p_model, sev_a, sev_b)

        prices_a, prices_b = prices_a or {}, prices_b or {}
        pin_a, pin_b = prices_a.get(SHARP_BOOK), prices_b.get(SHARP_BOOK)
        p_market = no_vig(pin_a, pin_b)
        market_source = "Pinnacle"
        if p_market is None and prices_a and prices_b:
            # No Pinnacle: use the consensus of the books we have (median no-vig).
            pairs = [no_vig(prices_a[k], prices_b[k]) for k in prices_a if k in prices_b]
            pairs = [p for p in pairs if p]
            if pairs:
                p_market, market_source = float(np.median(pairs)), "consensus"
        # Model only: the fair price is the model's own probability (after manual injury flags).
        fair = p_model_adj

        # Best price only among the books you can actually bet at (Settings.books).
        best_a, book_a = best_price(_allowed(prices_a, st.books))
        best_b, book_b = best_price(_allowed(prices_b, st.books))
        rec = None
        for side, p, price, book, name, mkt in (("A", fair, best_a, book_a, pa.name, p_market),
                                               ("B", 1 - fair, best_b, book_b, pb.name, None if p_market is None else 1 - p_market)):
            if not price:
                continue
            e = p * price - 1
            edge = (p - mkt) if mkt is not None else None
            # Needs a market reference (Pinnacle, else the consensus of the books) so the edge is known.
            ok = e >= st.min_ev and edge is not None and edge >= st.min_edge
            if ok and (rec is None or e > rec["ev"]):
                rec = {"market": "Moneyline", "selection": name, "side": side, "price": float(price), "book": book,
                       "fair_odds": 1 / p, "probability": p, "ev": float(e), "edge_vs_market": edge,
                       "stake": kelly_stake(p, price, st.kelly_fraction, st.kelly_cap)}

        result = {
            **base, "status": "ok", "event": ev.name, "event_key": ev.key, "level": ev.level, "surface": ev.surface,
            "indoor": ev.indoor, "best_of": ev.best_of,
            "p_model_a": p_model, "p_model_a_injury_adj": p_model_adj, "p_market_a": p_market, "market_source": market_source if p_market is not None else None,
            "p_fair_a": fair, "fair_odds_a": 1 / fair, "fair_odds_b": 1 / (1 - fair),
            "pinnacle_a": pin_a, "pinnacle_b": pin_b, "best_a": best_a, "best_book_a": book_a, "best_b": best_b, "best_book_b": book_b,
            "n_books": len(set(prices_a) & set(prices_b)),
            "court_speed": meta["speed"]["speed"], "court_speed_source": meta["speed"]["source"],
            "court_speed_prior": meta["speed"]["prior"], "court_speed_live_matches": meta["speed"]["live_matches"],
            "h2h": meta["h2h"], "h2h_surface": meta["h2h_surface"],
            "injury_flag_a": sev_a, "injury_flag_b": sev_b,
            "injury_signals_a": _injury_text(meta["a_state"]), "injury_signals_b": _injury_text(meta["b_state"]),
            "event_record_a": _event_text(meta["a_state"]), "event_record_b": _event_text(meta["b_state"]),
            "rank_a": meta["a_state"]["rank"], "rank_b": meta["b_state"]["rank"],
            "factors": self.explain(x), "recommendation": rec, "strategy": st.strategy,
            "stats_a": self.engine.profile(pa.pid, ev.surface), "stats_b": self.engine.profile(pb.pid, ev.surface),
            "prices_a": dict(prices_a), "prices_b": dict(prices_b),
        }
        if ev.level == "G" and ev.best_of == 5:
            result["sets"] = self._sets(p_market if p_market is not None else p_model_adj, meta, ctx, sets35 or {}, st)
        return result

    def _sets(self, p_a: float, meta: dict, ctx: dict, sets35: dict, st: Settings) -> dict:
        fav = max(p_a, 1 - p_a)
        p_over = predict_over35(fav, meta["a_state"], meta["b_state"], meta["speed"]["speed"], ROUND_INDEX.get(ctx["round"], 3))
        if p_over is None:
            return {"available": False}
        s = set_prob_from_match(p_a, 5)
        raw_over, raw_under = sets35.get("over", {}), sets35.get("under", {})
        over, under = _allowed(raw_over, st.books), _allowed(raw_under, st.books)
        best_o, book_o = best_price(over)
        best_u, book_u = best_price(under)
        rec = None
        for name, p, price, book in (("Over 3.5 sets", p_over, best_o, book_o), ("Under 3.5 sets", 1 - p_over, best_u, book_u)):
            if price and p * price - 1 >= st.min_ev_sets and (rec is None or p * price - 1 > rec["ev"]):
                rec = {"market": "Total sets 3.5", "selection": name, "price": float(price), "book": book, "probability": p,
                       "fair_odds": 1 / p, "ev": float(p * price - 1), "stake": kelly_stake(p, price, st.kelly_fraction, min(st.kelly_cap, 0.015))}
        return {
            "available": True, "p_over35": p_over, "fair_over": 1 / p_over, "fair_under": 1 / (1 - p_over),
            "best_over": best_o, "best_over_book": book_o, "best_under": best_u, "best_under_book": book_u,
            "pinnacle_over": raw_over.get(SHARP_BOOK), "pinnacle_under": raw_under.get(SHARP_BOOK),
            "independent_sets_view": sets_distribution(s), "recommendation": rec,
        }


def _round_code(name: str) -> str:
    n = str(name or "").upper().strip()
    for k in ROUND_INDEX:
        if n == k:
            return k
    words = {"1ST": "FIRST", "2ND": "SECOND", "3RD": "THIRD", "4TH": "FOURTH", "QUARTER": "QF", "SEMI": "SF", "FINAL": "F", "ROUND OF 16": "R16", "ROUND OF 32": "R32", "ROUND OF 64": "R64", "ROUND OF 128": "R128"}
    for k, v in words.items():
        if k in n:
            return v
    return n


def _injury_text(s: dict) -> str:
    bits = []
    if s["inj_recent"] > 0.25:
        bits.append("recent retirement/walkover")
    if s["inj_count"] >= 2:
        bits.append(f"{int(s['inj_count'])} retirements/walkovers in 12 months")
    if s["layoff"] > math.log1p(45):
        bits.append(f"{int(math.expm1(s['layoff']))} days since last match")
    if s["comeback"]:
        bits.append("early in a comeback")
    return "; ".join(bits)


def _event_text(s: dict) -> str:
    n = int(s["event_n"])
    if not n:
        return "first appearance at this event"
    return f"{n} matches here, {s['event_win']:.0%} shrunk win rate, {s['event_resid']:+.2f} vs rating"


# ---------------------------------------------------------------------------
# Slate
# ---------------------------------------------------------------------------
def player_key(name: str) -> str:
    t = normalize(name).split()
    return (t[0][:1] if t else "") + "|" + (t[-1] if t else "")


def same_player(a: str, b: str) -> bool:
    return player_key(a) == player_key(b)


def pair_key(a: str, b: str) -> tuple:
    return tuple(sorted((player_key(a), player_key(b))))


def build_slate(rt: Runtime, fixtures: list[dict], odds_events: list[dict], settings: Settings | None = None,
                now: float | None = None) -> dict:
    st = settings or Settings()
    now = now or time.time()
    by_pair: dict[tuple, dict] = {}
    for f in fixtures:
        by_pair[pair_key(f["player_a"], f["player_b"])] = dict(f)
    for o in odds_events:
        k = pair_key(o["player_a"], o["player_b"])
        cur = by_pair.get(k)
        if cur is None:
            by_pair[k] = dict(o)
            continue
        # align odds to the fixture's player order
        if same_player(cur["player_a"], o["player_a"]):
            cur["prices_a"], cur["prices_b"] = o.get("prices_a", {}), o.get("prices_b", {})
        else:
            cur["prices_a"], cur["prices_b"] = o.get("prices_b", {}), o.get("prices_a", {})
        cur["sets35"] = o.get("sets35", {})
        cur["odds_event_id"] = o.get("odds_event_id")
        if not cur.get("start_ts"):
            cur["start_ts"] = o.get("start_ts")
    preds, skipped = [], []
    for f in by_pair.values():
        ts = f.get("start_ts")
        if ts and (ts < now - 600 or ts > now + st.horizon_hours * 3600):
            continue
        status = str(f.get("status", "")).lower()
        if status and status not in {"not started", "upcoming", "scheduled", "notstarted", ""}:
            continue
        if re_qualifying(f.get("round", "")):
            continue
        p = rt.predict(f["player_a"], f["player_b"], f.get("tournament", ""), ts, f.get("round", ""),
                       f.get("prices_a"), f.get("prices_b"), f.get("sets35"), st)
        p["fixture_id"] = f.get("id") or f.get("odds_event_id")
        (preds if p["status"] == "ok" else skipped).append(p)
    preds.sort(key=lambda r: (r.get("start_ts") or 0))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_at_local": datetime.now(ZoneInfo(APP_TIMEZONE)).strftime("%a %b %d, %H:%M"),
        "settings": st.__dict__, "matches": preds, "skipped": skipped,
        "model": {"latest_match_date": rt.metrics.get("latest_match_date"), "blend": rt.model.blend},
    }


def re_qualifying(name: str) -> bool:
    import re
    return bool(re.search(r"(qualif|^q\d$)", str(name or ""), re.I))


LATEST_PATH = GENERATED.parent / "predictions" / "latest.json"
