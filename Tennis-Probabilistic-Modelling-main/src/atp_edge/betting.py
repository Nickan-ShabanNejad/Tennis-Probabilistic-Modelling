"""Prices, fair probabilities, expected value and stake sizing."""
from __future__ import annotations

import math

import numpy as np

MIN_P, MAX_P = 0.01, 0.99


def valid_odds(v) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) and f > 1.0 else None


def no_vig(odds_a, odds_b) -> float | None:
    """Fair probability of side A from a two-way price, bookmaker margin removed.

    Uses the "power" method: find k so that (1/a)^k + (1/b)^k = 1. Bookmakers load more of
    their margin onto outsiders (favourite-longshot bias); simply rescaling the two implied
    probabilities therefore rates outsiders too highly. On 37,683 Pinnacle closing prices
    (2010-2026) the power method has the lower log loss (0.5686 vs 0.5690), and on the
    books' average price the gap is three times larger.
    """
    a, b = valid_odds(odds_a), valid_odds(odds_b)
    if a is None or b is None:
        return None
    return float(no_vig_array([a], [b])[0])


def no_vig_array(odds_a, odds_b) -> np.ndarray:
    """Vectorised ``no_vig``; NaN where either price is missing or not above 1."""
    a, b = np.asarray(odds_a, float), np.asarray(odds_b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 1.0) & (b > 1.0)
    ia, ib = 1 / np.where(ok, a, 2.0), 1 / np.where(ok, b, 2.0)
    lo, hi = np.full(ia.shape, 0.05), np.full(ia.shape, 20.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_high = ia ** mid + ib ** mid > 1
        lo, hi = np.where(too_high, mid, lo), np.where(too_high, hi, mid)
    return np.where(ok, ia ** ((lo + hi) / 2), np.nan)


def no_vig_proportional(odds_a, odds_b) -> float | None:
    """The simple rescaling method (kept for comparison in the model report)."""
    a, b = valid_odds(odds_a), valid_odds(odds_b)
    if a is None or b is None:
        return None
    return (1 / a) / (1 / a + 1 / b)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def blend(p_model: float, p_market: float | None, coef: dict | None) -> float:
    """Benter-style: logit(p) = a*logit(model) + b*logit(market)."""
    if p_market is None or not coef:
        return float(p_model)
    a, b = float(coef.get("model_coef", 0.0)), float(coef.get("market_coef", 1.0))
    return float(np.clip(sigmoid(a * logit(p_model) + b * logit(p_market)), MIN_P, MAX_P))


def injury_adjust(p_a: float, sev_a: float, sev_b: float, strength: float = 1.2) -> float:
    """Manual injury flags (0 = fit, 1 = seriously hampered) shift the log-odds.

    This is a judgment input, not a learned one: severity 0.5 on one player moves
    a 50/50 match to about 35/65. Set flags in data/reference/injuries.csv.
    """
    if not sev_a and not sev_b:
        return float(p_a)
    return float(np.clip(sigmoid(logit(p_a) - strength * (float(sev_a) - float(sev_b))), MIN_P, MAX_P))


def ev(p: float, odds: float) -> float:
    return float(p) * float(odds) - 1.0


def kelly_stake(p: float, odds: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    o = valid_odds(odds)
    if o is None:
        return 0.0
    full = max(0.0, (p * o - 1.0) / (o - 1.0))
    return float(min(full * fraction, cap))


# ---------------------------------------------------------------------------
# Best-of-five helpers for the Grand Slam total-sets market
# ---------------------------------------------------------------------------
def bo3_win(s: float) -> float:
    return s * s * (3 - 2 * s)


def bo5_win(s: float) -> float:
    q = 1 - s
    return s ** 3 * (1 + 3 * q + 6 * q * q)


def set_prob_from_match(p_match: float, best_of: int = 5) -> float:
    """Invert P(win match) to the per-set win probability (bisection)."""
    f = bo5_win if best_of == 5 else bo3_win
    lo, hi = 1e-6, 1 - 1e-6
    target = float(np.clip(p_match, 1e-6, 1 - 1e-6))
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def over35_from_set_prob(s: float) -> float:
    """P(at least 4 sets in best-of-5) assuming independent sets."""
    return 1.0 - s ** 3 - (1 - s) ** 3


def sets_distribution(s: float) -> dict:
    q = 1 - s
    return {
        "A 3-0": s ** 3, "A 3-1": 3 * s ** 3 * q, "A 3-2": 6 * s ** 3 * q * q,
        "B 3-0": q ** 3, "B 3-1": 3 * q ** 3 * s, "B 3-2": 6 * q ** 3 * s * s,
    }


def backtest(p_winner, odds_w, odds_l, *, min_ev=0.02, min_edge=None, fraction=0.25, cap=0.02, market_w=None,
             max_odds=None, stake_p_winner=None, min_stake_ev=0.0) -> dict:
    """Replay a rule on finished matches (rows in date order).

    ``p_winner`` is the probability the rule would have used for the eventual winner to
    decide whether a price is a bet: it backs whichever side shows the higher EV at the given
    prices, if that EV is at least ``min_ev``.
    ``min_edge`` (optional) also requires that probability to exceed the market's by that much.
    ``stake_p_winner`` (optional) is a second probability used for the stake; the bet is
    skipped when its EV is below ``min_stake_ev`` (the "second opinion must not disagree" rule).
    Profit is in units of the starting bankroll with fractional-Kelly stakes (not compounded).
    """
    p = np.asarray(p_winner, float)
    ow, ol = np.asarray(odds_w, float), np.asarray(odds_l, float)
    mw = (1 / ow) / (1 / ow + 1 / ol) if market_w is None else np.asarray(market_w, float)
    ev_w, ev_l = p * ow - 1, (1 - p) * ol - 1
    back_w = ev_w >= ev_l
    e = np.where(back_w, ev_w, ev_l)
    edge = np.where(back_w, p - mw, (1 - p) - (1 - mw))
    price = np.where(back_w, ow, ol)
    sel = (e >= min_ev) & np.isfinite(e)
    if min_edge is not None:      # only the model-only rule asks for a gap to the market
        sel &= edge >= min_edge
    if max_odds is not None:
        sel &= price <= max_odds
    stake_ev = e
    if stake_p_winner is not None:
        sp = np.asarray(stake_p_winner, float)
        stake_ev = np.where(back_w, sp * ow - 1, (1 - sp) * ol - 1)
        sel &= np.isfinite(stake_ev) & (stake_ev >= min_stake_ev)
    if not sel.any():
        return {"bets": 0}
    stake = np.minimum(fraction * np.maximum(stake_ev, 0) / np.maximum(price - 1, 1e-9), cap)[sel]
    pl = np.where(back_w, price - 1, -1.0)[sel]
    curve = np.cumsum(stake * pl)
    return {
        "bets": int(sel.sum()), "share_of_matches": float(sel.mean()), "win_rate": float(back_w[sel].mean()),
        "avg_odds": float(price[sel].mean()), "avg_claimed_ev": float(stake_ev[sel].mean()),
        "flat_roi": float(pl.mean()), "flat_roi_se": float(pl.std(ddof=1) / np.sqrt(len(pl))) if len(pl) > 1 else float("nan"),
        "kelly_roi": float((stake * pl).sum() / stake.sum()) if stake.sum() > 0 else float("nan"),
        "profit_units": float(curve[-1]), "staked_units": float(stake.sum()),
        "max_drawdown_units": float(np.max(np.maximum.accumulate(np.r_[0.0, curve]) - np.r_[0.0, curve])),
    }
