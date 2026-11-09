"""Chronological rating & feature engine.

ONE code path builds features for training and for live predictions:

    engine = Engine(court_speed)
    for match in matches (oldest first):
        x = engine.features(a, b, context)   # uses only information before the match
        engine.update(match)                 # then learns from it

After the full history has been processed the same ``engine`` object is saved
and ``engine.features`` is called for tomorrow's matches. There is no separate
"live feature" implementation that could drift from training.

Feature families
----------------
ratings      dynamic-K Elo (overall + surface + blend), margin-of-victory weighted
serve/return surface-specific EWMA of service/return points won, recent windows, ace rate
form         last-10 win rate, decayed form, Elo trend, opponent quality
fatigue      matches / sets / minutes in the last 7-14 days, rest days
injury       retirements & walkovers (recency-weighted), layoff length, comeback
             matches, plus manual injury flags (data/reference/injuries.csv)
event        player's results at THIS tournament vs what his rating predicted
event type   same for the tournament level (Slam / Masters / 500 / 250 / Finals)
             and for best-of-five
court speed  Court Speed Index for this event (Tennis Abstract + in-event
             estimate), the player's measured speed preference, and speed x
             serve/return/ace interactions
h2h          shrunk head-to-head record, overall and on the surface
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from .court_speed import CourtSpeed, blend_live
from .tournaments import canonical, encode_level, normalize

RET_MARKERS = ("RET", "W/O", "WO", "DEF", "ABN", "ABD", "WALKOVER", "UNFINISHED")
ROUND_INDEX = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 4, "BR": 6,
               "FIRST": 2, "SECOND": 3, "THIRD": 4, "FOURTH": 5, "1/4": 5, "1/2": 6, "FINAL": 7}

# Rating constants (FiveThirtyEight-style dynamic K). Tuned on 2005-2013.
K_C, K_OFFSET, K_SHAPE = 150.0, 5.0, 0.3
MOV_WEIGHT = 1.0
LEVEL_K = {"G": 1.1, "F": 1.05, "D": 0.85, "A": 0.9, "C": 0.7}   # Challengers move ratings less
EWMA_ALPHA = 0.15
EVENT_PRIOR, LEVEL_PRIOR, BO5_PRIOR, H2H_PRIOR = 6.0, 20.0, 12.0, 6.0
SPEED_RIDGE, SPEED_PRIOR = 0.35, 12.0


def parse_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(int(float(value))) if not isinstance(value, str) else value.replace("-", "")[:8]
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def score_info(score) -> dict:
    """Games won/lost by the winner, sets, and whether the match was completed."""
    text = str(score or "").upper()
    incomplete = any(m in text for m in RET_MARKERS)
    walkover = "W/O" in text or "WALKOVER" in text or text.strip() in {"WO", "W O"}
    gw = gl = sets_w = sets_l = 0
    for a, b in re.findall(r"(\d{1,2})-(\d{1,2})", text):
        a, b = int(a), int(b)
        if a > 7 or b > 7:   # match tiebreak points
            if a > b:
                sets_w += 1
            else:
                sets_l += 1
            continue
        gw += a
        gl += b
        if (a >= 6 or b >= 6) and a != b:
            if a > b:
                sets_w += 1
            else:
                sets_l += 1
    return {"incomplete": incomplete, "walkover": walkover, "games_w": gw, "games_l": gl,
            "sets": sets_w + sets_l, "sets_w": sets_w, "sets_l": sets_l}


def _is_indoor(v) -> bool:
    return str(v or "").strip().upper() in {"I", "1", "1.0", "TRUE", "INDOOR"}


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _ratio(n, d):
    n, d = _num(n), _num(d)
    return n / d if n is not None and d and d > 0 else None


def _k(n: int) -> float:
    return K_C / ((n + K_OFFSET) ** K_SHAPE)


@dataclass
class Player:
    pid: str
    name: str = ""
    elo: float = 1500.0
    n: int = 0
    surf_elo: dict = field(default_factory=dict)
    surf_n: dict = field(default_factory=dict)
    serve: dict = field(default_factory=dict)     # surface -> EWMA service points won
    ret: dict = field(default_factory=dict)       # surface -> EWMA return points won
    hist: deque = field(default_factory=lambda: deque(maxlen=30))
    elo_trail: deque = field(default_factory=lambda: deque(maxlen=11))
    last_date: date | None = None
    injury_dates: list = field(default_factory=list)   # retirements / walkovers given
    return_date: date | None = None                     # first match after a 60+ day layoff
    matches_since_return: int = 0
    event_perf: dict = field(default_factory=dict)      # event -> [sum residual, n, wins]
    level_perf: dict = field(default_factory=dict)      # level -> [sum, n]
    bo5: list = field(default_factory=lambda: [0.0, 0])
    speed: list = field(default_factory=lambda: [0.0, 0.0, 0])  # xy, xx, n
    slam_sets: list = field(default_factory=lambda: [0, 0, 0])  # n, straight, 4+ sets
    rank: float = 500.0
    rank_points: float = 0.0
    birth_year: float | None = None
    height: float | None = None
    hand: str = ""


class Engine:
    def __init__(self, court_speed: CourtSpeed | None = None):
        self.cs = court_speed or CourtSpeed()
        self.players: dict[str, Player] = {}
        self.name_index: dict[str, str] = {}
        self.h2h: dict[tuple[str, str], list] = {}
        self.event_live: dict[tuple, list] = {}       # (season, event, surface) -> [sum ratio, n]
        self.event_emp: dict[tuple, dict] = {}        # (event, surface) -> {season: [sum, n]} empirical speed
        self.manual_injuries: dict[str, float] = {}   # normalized name -> severity 0..1
        self.last_date: date | None = None

    # ------------------------------------------------------------------ lookup
    def player(self, pid: str, name: str = "") -> Player:
        p = self.players.get(pid)
        if p is None:
            p = Player(pid=pid, name=name)
            self.players[pid] = p
        if name:
            p.name = name
            self.name_index[normalize(name)] = pid
        return p

    def find(self, name: str) -> Player | None:
        key = normalize(name)
        pid = self.name_index.get(key)
        if pid:
            return self.players[pid]
        # loose: same surname + first initial, unique among recently active players
        toks = key.split()
        if len(toks) >= 2:
            cands = [self.players[v] for k, v in self.name_index.items()
                     if k.split() and k.split()[0][:1] == toks[0][:1] and k.split()[-1] == toks[-1]]
            cands = {c.pid: c for c in cands}
            if len(cands) == 1:
                return next(iter(cands.values()))
            if len(cands) > 1:  # prefer the most recently active
                return max(cands.values(), key=lambda c: c.last_date or date(1900, 1, 1))
        return None

    # ------------------------------------------------------------ court speed
    def speed_for(self, tournament: str, surface: str, season: int) -> dict:
        key = canonical(tournament)
        prior, n_seasons = self.cs.prior(tournament, surface, season)
        source = "tennis_abstract" if n_seasons else "surface_default"
        if not n_seasons:
            emp = [(s, v[0] / v[1]) for s, v in self.event_emp.get((key, surface), {}).items()
                   if s < season and v[1] >= 5]
            if emp:
                emp = sorted(emp)[-3:]
                prior = float(np.mean([v for _, v in emp]))
                source = "match_stats_history"
        live = self.event_live.get((season, key, surface))
        live_val = live[0] / live[1] if live and live[1] else None
        speed, w = blend_live(prior, live_val, live[1] if live else 0)
        return {"speed": speed, "prior": prior, "live": live_val, "live_matches": live[1] if live else 0,
                "live_weight": w, "source": source if w == 0 else source + "+live"}

    # --------------------------------------------------------------- states
    def _window(self, p: Player, key: str, n: int, default: float, surface: str | None = None) -> float:
        vals = [m[key] for m in list(p.hist)[::-1] if (surface is None or m["surface"] == surface) and m.get(key) is not None][:n]
        return float(np.mean(vals)) if vals else default

    def state(self, p: Player, surface: str, when: date, level: str, best_of: int, event: str, speed: float,
              indoor: bool = False) -> dict:
        hist = list(p.hist)
        s_elo = p.surf_elo.get(surface, 1500.0)
        days_since = max((when - p.last_date).days, 0) if p.last_date else 365
        inj = [d for d in p.injury_dates if 0 <= (when - d).days <= 365]
        inj_recent = sum(math.exp(-(when - d).days / 45.0) for d in inj)
        rest = min(max(days_since, 0), 60)
        last7 = [m for m in hist if 0 <= (when - m["date"]).days <= 7]
        last14 = [m for m in hist if 0 <= (when - m["date"]).days <= 14]
        ev = p.event_perf.get(event, [0.0, 0, 0])
        lv = p.level_perf.get(level, [0.0, 0])
        sx = p.speed
        slope = (sx[0] / (sx[1] + SPEED_RIDGE)) * (sx[2] / (sx[2] + SPEED_PRIOR)) if sx[2] else 0.0
        trail = list(p.elo_trail)
        age = (when.year - p.birth_year) if p.birth_year else 26.0
        return {
            "elo": p.elo, "surf_elo": s_elo, "blend_elo": 0.5 * p.elo + 0.5 * s_elo,
            "cond_elo": p.surf_elo.get(("Hard-I" if indoor else "Hard-O") if surface == "Hard" else surface, s_elo),
            "exp": math.log1p(p.n), "surf_exp": math.log1p(p.surf_n.get(surface, 0)),
            "serve": p.serve.get(surface, p.serve.get("all", 0.63)),
            "ret": p.ret.get(surface, p.ret.get("all", 0.37)),
            "spw10": self._window(p, "spw", 10, 0.63), "rpw10": self._window(p, "rpw", 10, 0.37),
            "spw_adj10": self._window(p, "spw_adj", 10, 0.63), "rpw_adj10": self._window(p, "rpw_adj", 10, 0.37),
            "ace10": self._window(p, "ace", 10, 0.08), "df10": self._window(p, "df", 10, 0.035),
            "win10": self._window(p, "win", 10, 0.5), "surf_win10": self._window(p, "win", 10, 0.5, surface),
            "form": float(np.average([m["win"] for m in hist[-10:]], weights=[0.8 ** (len(hist[-10:]) - 1 - i) for i in range(len(hist[-10:]))])) if hist else 0.5,
            "opp10": self._window(p, "opp_elo", 10, 1500.0),
            "elo_trend": (p.elo - trail[0]) if trail else 0.0,
            "m7": float(len(last7)), "m14": float(len(last14)),
            "sets7": float(sum(m["sets"] for m in last7)),
            "min7": float(sum(m["minutes"] or 0 for m in last7)),
            "rest": float(rest),
            "inj_recent": float(inj_recent), "inj_count": float(len(inj)),
            "layoff": math.log1p(min(days_since, 365)),
            "comeback": 1.0 if (p.return_date and p.matches_since_return < 4) else 0.0,
            "manual_inj": float(self.manual_injuries.get(normalize(p.name), 0.0)),
            "event_resid": ev[0] / (ev[1] + EVENT_PRIOR) if ev[1] else 0.0,
            "event_win": (ev[2] + 0.5 * EVENT_PRIOR) / (ev[1] + EVENT_PRIOR),
            "event_n": float(ev[1]),
            "level_resid": lv[0] / (lv[1] + LEVEL_PRIOR) if lv[1] else 0.0,
            "level_n": float(lv[1]),
            "bo5_resid": (p.bo5[0] / (p.bo5[1] + BO5_PRIOR)) if (best_of == 5 and p.bo5[1]) else 0.0,
            "speed_slope": slope, "speed_n": float(sx[2]),
            "rank": max(float(p.rank or 500.0), 1.0), "rank_points": float(p.rank_points or 0.0),
            "age": float(age), "height": float(p.height or 185.0),
            "slam_n": float(p.slam_sets[0]),
            "slam_straight": (p.slam_sets[1] + 0.45 * 10) / (p.slam_sets[0] + 10),
            "slam_long": (p.slam_sets[2] + 0.55 * 10) / (p.slam_sets[0] + 10),
        }

    # ------------------------------------------------------------- features
    def features(self, a_id: str, b_id: str, ctx: dict) -> dict:
        """ctx: tournament, surface, level, best_of, indoor, round, date."""
        when = parse_date(ctx["date"])
        surface, level, best_of = ctx["surface"], str(ctx.get("level", "250")), int(ctx.get("best_of") or 3)
        event = canonical(ctx.get("tournament", ""))
        sp = ctx.get("speed_info") or self.speed_for(ctx.get("tournament", ""), surface, when.year)
        speed = float(sp["speed"])
        pa = self.players.get(a_id) or Player(pid=a_id)
        pb = self.players.get(b_id) or Player(pid=b_id)
        indoor = bool(ctx.get("indoor"))
        A = self.state(pa, surface, when, level, best_of, event, speed, indoor)
        B = self.state(pb, surface, when, level, best_of, event, speed, indoor)
        d = {k: A[k] - B[k] for k in A}
        pair = tuple(sorted((a_id, b_id)))
        h = self.h2h.get(pair, [])
        aw = sum(1 for m in h if m[0] == a_id)
        hs = [m for m in h if m[1] == surface]
        asw = sum(1 for m in hs if m[0] == a_id)
        sc = speed - 1.0
        lvl = encode_level(level)
        blend = d["blend_elo"]
        f = {
            # ratings
            "elo_diff": d["elo"], "surf_elo_diff": d["surf_elo"], "blend_elo_diff": blend, "cond_elo_diff": d["cond_elo"],
            "elo_prob_logit": blend * math.log(10) / 400.0,
            "exp_diff": d["exp"], "surf_exp_diff": d["surf_exp"],
            # serve / return
            "serve_diff": d["serve"], "return_diff": d["ret"], "spw10_diff": d["spw10"], "rpw10_diff": d["rpw10"],
            "spw_adj10_diff": d["spw_adj10"], "rpw_adj10_diff": d["rpw_adj10"],
            "ace10_diff": d["ace10"], "df10_diff": d["df10"],
            # form
            "win10_diff": d["win10"], "surf_win10_diff": d["surf_win10"], "form_diff": d["form"],
            "opp10_diff": d["opp10"] / 400.0, "elo_trend_diff": d["elo_trend"],
            # fatigue
            "m7_diff": d["m7"], "m14_diff": d["m14"], "sets7_diff": d["sets7"], "min7_diff": d["min7"] / 60.0,
            "rest_diff": d["rest"],
            # injury
            "inj_recent_diff": d["inj_recent"], "inj_count_diff": d["inj_count"], "layoff_diff": d["layoff"],
            "comeback_diff": d["comeback"],
            # event history & event type
            "event_resid_diff": d["event_resid"], "event_win_diff": d["event_win"],
            "level_resid_diff": d["level_resid"], "bo5_resid_diff": d["bo5_resid"],
            # court speed
            "speed_fit_diff": d["speed_slope"] * sc, "speed_serve": sc * d["serve"] * 10,
            "speed_return": sc * d["ret"] * 10, "speed_ace": sc * d["ace10"] * 10, "speed_elo": sc * blend / 400.0,
            # head to head (shrunk)
            "h2h_edge": (2 * aw - len(h)) / (len(h) + H2H_PRIOR) if h else 0.0,
            "h2h_surf_edge": (2 * asw - len(hs)) / (len(hs) + H2H_PRIOR) if hs else 0.0,
            # ranking / physical
            "log_rank_diff": math.log(B["rank"]) - math.log(A["rank"]),
            "age_diff": d["age"], "height_diff": d["height"] / 10.0,
            # interactions
            "level_elo": (lvl - 2.0) * blend / 400.0, "bo5_elo": (best_of - 3) * blend / 400.0,
            "indoor_serve": float(bool(ctx.get("indoor"))) * d["serve"] * 10,
            # context (not negated when players are swapped)
            "ctx_speed": speed, "ctx_level": lvl, "ctx_best_of": float(best_of),
            "ctx_indoor": float(bool(ctx.get("indoor"))), "ctx_round": float(ROUND_INDEX.get(str(ctx.get("round", "")).upper(), 3)),
            "ctx_h2h_n": math.log1p(len(h)), "ctx_event_n": math.log1p(min(A["event_n"], B["event_n"])),
        }
        meta = {"speed": sp, "a_state": A, "b_state": B, "h2h": (aw, len(h) - aw), "h2h_surface": (asw, len(hs) - asw)}
        return {"x": f, "meta": meta}

    # --------------------------------------------------------------- update
    def update(self, r: dict, a_state_cache: dict | None = None) -> None:
        when = parse_date(r["tourney_date"])
        self.last_date = when if self.last_date is None or when > self.last_date else self.last_date
        surface = str(r.get("surface", "Hard")).title()
        level = str(r.get("tourney_level", "250"))
        best_of = int(_num(r.get("best_of")) or (5 if level == "G" else 3))
        event = canonical(r.get("tourney_name", ""))
        w = self.player(str(r["winner_id"]), str(r.get("winner_name") or ""))
        l = self.player(str(r["loser_id"]), str(r.get("loser_name") or ""))
        for p, side in ((w, "winner"), (l, "loser")):
            rk = _num(r.get(f"{side}_rank"))
            if rk:
                p.rank = rk
            pts = _num(r.get(f"{side}_rank_points"))
            if pts:
                p.rank_points = pts
            age = _num(r.get(f"{side}_age"))
            if age:
                p.birth_year = when.year - age
            ht = _num(r.get(f"{side}_ht"))
            if ht:
                p.height = ht
        info = score_info(r.get("score"))
        if info["incomplete"]:
            # The loser retired, defaulted or withdrew: an injury signal, not a result.
            l.injury_dates.append(when)
            l.injury_dates = l.injury_dates[-10:]
            if not info["walkover"]:
                for p in (w, l):
                    self._touch(p, when)
            return

        # ---- ratings
        share = info["games_w"] / (info["games_w"] + info["games_l"]) if info["games_w"] + info["games_l"] else None
        mult = (1.0 + MOV_WEIGHT * (2 * share - 1)) if share is not None else 1.0
        mult *= LEVEL_K.get(level, 1.0)
        exp = 1 / (1 + 10 ** ((l.elo - w.elo) / 400))
        ws, ls = w.surf_elo.get(surface, 1500.0), l.surf_elo.get(surface, 1500.0)
        sexp = 1 / (1 + 10 ** ((ls - ws) / 400))
        blend_exp = 1 / (1 + 10 ** (((0.5 * l.elo + 0.5 * ls) - (0.5 * w.elo + 0.5 * ws)) / 400))
        pre_w_elo, pre_l_elo = w.elo, l.elo
        w.elo += _k(w.n) * (1 - exp) * mult
        l.elo -= _k(l.n) * (1 - exp) * mult
        w.surf_elo[surface] = ws + _k(w.surf_n.get(surface, 0)) * (1 - sexp) * mult
        l.surf_elo[surface] = ls - _k(l.surf_n.get(surface, 0)) * (1 - sexp) * mult
        # Condition rating: hard courts split into indoor and outdoor (different balls, no wind/sun).
        if surface == "Hard":
            cond = "Hard-I" if _is_indoor(r.get("indoor")) else "Hard-O"
            wc, lc = w.surf_elo.get(cond, ws), l.surf_elo.get(cond, ls)
            cexp = 1 / (1 + 10 ** ((lc - wc) / 400))
            w.surf_elo[cond] = wc + _k(w.surf_n.get(cond, 0)) * (1 - cexp) * mult
            l.surf_elo[cond] = lc - _k(l.surf_n.get(cond, 0)) * (1 - cexp) * mult
            for p in (w, l):
                p.surf_n[cond] = p.surf_n.get(cond, 0) + 1
        for p in (w, l):
            p.n += 1
            p.surf_n[surface] = p.surf_n.get(surface, 0) + 1
        w.elo_trail.append(pre_w_elo)
        l.elo_trail.append(pre_l_elo)

        # ---- serve / return stats
        stats = {}
        for side, opp in (("w", "l"), ("l", "w")):
            svpt = _num(r.get(f"{side}_svpt"))
            won = (_num(r.get(f"{side}_1stWon")) or 0) + (_num(r.get(f"{side}_2ndWon")) or 0) if _num(r.get(f"{side}_1stWon")) is not None else None
            osv = _num(r.get(f"{opp}_svpt"))
            owon = (_num(r.get(f"{opp}_1stWon")) or 0) + (_num(r.get(f"{opp}_2ndWon")) or 0) if _num(r.get(f"{opp}_1stWon")) is not None else None
            first_in = _num(r.get(f"{side}_1stIn"))
            bp_faced, bp_saved = _num(r.get(f"{side}_bpFaced")), _num(r.get(f"{side}_bpSaved"))
            obp_faced, obp_saved = _num(r.get(f"{opp}_bpFaced")), _num(r.get(f"{opp}_bpSaved"))
            pts = (svpt + osv) if (svpt and osv) else None
            stats[side] = {
                "spw": _ratio(won, svpt), "rpw": (1 - owon / osv) if (owon is not None and osv) else None,
                "ace": _ratio(r.get(f"{side}_ace"), svpt), "df": _ratio(r.get(f"{side}_df"), svpt), "svpt": svpt,
                # display-only statistics (match page comparison)
                "first_in": _ratio(first_in, svpt), "first_won": _ratio(r.get(f"{side}_1stWon"), first_in),
                "second_won": _ratio(r.get(f"{side}_2ndWon"), (svpt - first_in) if (svpt and first_in is not None) else None),
                "bp_save": _ratio(bp_saved, bp_faced),
                "bp_conv": _ratio((obp_faced - obp_saved) if (obp_faced is not None and obp_saved is not None) else None, obp_faced),
                "winners": _ratio(r.get(f"{side}_winners"), pts), "ue": _ratio(r.get(f"{side}_unforced_errors"), pts),
            }
        minutes = _num(r.get("minutes"))
        speed_info = self.speed_for(r.get("tourney_name", ""), surface, when.year)
        # Opponent-adjusted serve/return: credit for holding serve against a strong returner and
        # for winning return points against a big server (pre-match opponent ratings).
        pre_ret = {pid: pl.ret.get(surface, pl.ret.get("all")) for pid, pl in ((w.pid, w), (l.pid, l))}
        pre_srv = {pid: pl.serve.get(surface, pl.serve.get("all")) for pid, pl in ((w.pid, w), (l.pid, l))}
        for p, side, won, opp_elo, resid in ((w, "w", 1.0, pre_l_elo, 1 - blend_exp), (l, "l", 0.0, pre_w_elo, -(1 - blend_exp))):
            s = stats[side]
            opp = l if p is w else w
            s["spw_adj"] = (s["spw"] + ((pre_ret[opp.pid] or 0.37) - 0.37)) if s["spw"] is not None else None
            s["rpw_adj"] = (s["rpw"] + ((pre_srv[opp.pid] or 0.63) - 0.63)) if s["rpw"] is not None else None
            for key in (surface, "all"):
                if s["spw"] is not None:
                    p.serve[key] = s["spw"] if key not in p.serve else EWMA_ALPHA * s["spw"] + (1 - EWMA_ALPHA) * p.serve[key]
                if s["rpw"] is not None:
                    p.ret[key] = s["rpw"] if key not in p.ret else EWMA_ALPHA * s["rpw"] + (1 - EWMA_ALPHA) * p.ret[key]
            p.hist.append({"date": when, "surface": surface, "win": won, "opp_elo": opp_elo, "sets": info["sets"],
                           "minutes": minutes, **{k: s[k] for k in ("spw", "rpw", "spw_adj", "rpw_adj", "ace", "df", "first_in", "first_won",
                                                                    "second_won", "bp_save", "bp_conv", "winners", "ue")}})
            ev = p.event_perf.setdefault(event, [0.0, 0, 0])
            ev[0] += resid; ev[1] += 1; ev[2] += int(won)
            lv = p.level_perf.setdefault(level, [0.0, 0])
            lv[0] += resid; lv[1] += 1
            if best_of == 5:
                p.bo5[0] += resid; p.bo5[1] += 1
                if level == "G" and info["sets"] >= 3:
                    p.slam_sets[0] += 1
                    p.slam_sets[1] += int(info["sets"] == 3)
                    p.slam_sets[2] += int(info["sets"] >= 4)
            sx = speed_info["speed"] - 1.0
            p.speed[0] += sx * resid; p.speed[1] += sx * sx; p.speed[2] += 1
            self._touch(p, when)

        pair = tuple(sorted((w.pid, l.pid)))
        self.h2h.setdefault(pair, []).append((w.pid, surface))
        self.h2h[pair] = self.h2h[pair][-30:]

        # ---- in-event court speed (ace rate vs what these two servers usually produce)
        ws_, ls_ = stats["w"], stats["l"]
        if ws_["svpt"] and ls_["svpt"] and ws_["ace"] is not None and ls_["ace"] is not None:
            exp_w = self._window(w, "ace", 10, 0.08) if len(w.hist) > 3 else 0.08
            exp_l = self._window(l, "ace", 10, 0.08) if len(l.hist) > 3 else 0.08
            expected = (exp_w * ws_["svpt"] + exp_l * ls_["svpt"]) / (ws_["svpt"] + ls_["svpt"])
            actual = (ws_["ace"] * ws_["svpt"] + ls_["ace"] * ls_["svpt"]) / (ws_["svpt"] + ls_["svpt"])
            if expected > 0.005:
                ratio = float(np.clip(actual / expected, 0.3, 3.0))
                live = self.event_live.setdefault((when.year, event, surface), [0.0, 0])
                live[0] += ratio; live[1] += 1
                emp = self.event_emp.setdefault((event, surface), {}).setdefault(when.year, [0.0, 0])
                emp[0] += ratio; emp[1] += 1

    def _touch(self, p: Player, when: date) -> None:
        if p.last_date and (when - p.last_date).days > 60:
            p.return_date, p.matches_since_return = when, 0
        elif p.return_date:
            p.matches_since_return += 1
        p.last_date = when

    # ----------------------------------------------------------- display
    def profile(self, pid: str, surface: str) -> dict:
        """Side-by-side statistics for the match page (not model features)."""
        p = self.players.get(pid)
        if p is None:
            return {}

        def w(key, n=10, surf=None):
            vals = [m.get(key) for m in list(p.hist)[::-1] if (surf is None or m["surface"] == surf) and m.get(key) is not None][:n]
            return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

        out = {"Elo (overall)": p.elo, f"Elo ({surface})": p.surf_elo.get(surface), "ATP rank": p.rank,
               "Matches played (career, in database)": p.n,
               "Win rate, last 10": w("win")[0], f"Win rate, last 10 on {surface}": w("win", 10, surface)[0]}
        for key, label in (("spw", "Service points won"), ("rpw", "Return points won"), ("first_in", "1st serve in"),
                           ("first_won", "1st serve points won"), ("second_won", "2nd serve points won"),
                           ("ace", "Aces per service point"), ("df", "Double faults per service point"),
                           ("bp_save", "Break points saved"), ("bp_conv", "Break points converted"),
                           ("winners", "Winners per point played"), ("ue", "Unforced errors per point played")):
            val, n = w(key)
            out[f"{label} (last {n})" if n else label] = val
        return out

    # -------------------------------------------------------------- helpers
    def apply_rankings(self, rankings) -> int:
        """rankings: iterable of (name, rank, points). Returns players matched."""
        hit = 0
        for name, rank, points in rankings:
            p = self.find(name)
            if p is not None:
                p.rank, p.rank_points = float(rank), float(points or 0)
                hit += 1
        return hit

    def set_manual_injuries(self, rows) -> None:
        """rows: iterable of (name, severity 0..1, until_date or None)."""
        today = date.today()
        out = {}
        for name, sev, until in rows:
            try:
                if until and parse_date(until) < today:
                    continue
            except Exception:
                pass
            out[normalize(name)] = float(sev)
        self.manual_injuries = out


DIFF_PREFIX_EXCLUDE = ("ctx_",)


def feature_names(sample: dict) -> list[str]:
    return list(sample.keys())


def is_context(name: str) -> bool:
    return name.startswith("ctx_")
