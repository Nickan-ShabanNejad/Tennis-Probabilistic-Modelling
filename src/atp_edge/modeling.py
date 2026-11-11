"""Model training, walk-forward evaluation, market benchmark and the Slam sets model."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .betting import backtest, logit, no_vig_array, over35_from_set_prob, set_prob_from_match, sigmoid
from .config import GENERATED, MODEL_DIR
from .odds_history import attach, load_odds

MATCH_MODEL = MODEL_DIR / "match_model.joblib"
SETS_MODEL = MODEL_DIR / "sets_model.joblib"
ENGINE_PATH = MODEL_DIR / "engine.joblib"
METRICS_PATH = GENERATED / "metrics.json"

CANDIDATES = {
    "logistic_regression": Pipeline([
        ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.05, max_iter=4000)),
    ]),
    "gradient_boosting": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
                                                 min_samples_leaf=100, l2_regularization=3.0, random_state=1)),
    ]),
}


# Recent seasons count more when the model is fitted: the game changes (serve speeds, the
# balls, how young players develop). A match from 5 seasons ago gets half the weight of
# one from this season. Walk-forward log loss 2018-2026: 0.5945 unweighted, 0.5937 weighted.
RECENCY_HALF_LIFE_YEARS = 5.0


def recency_weights(years, reference_year) -> np.ndarray:
    return 0.5 ** ((float(reference_year) - np.asarray(years, float)) / RECENCY_HALF_LIFE_YEARS)


class MatchModel:
    """Symmetric wrapper: P(A beats B) = 1 - P(B beats A) by construction."""

    def __init__(self, features: list[str], pipeline=None, name: str = ""):
        self.features = list(features)
        self.diff_idx = [i for i, f in enumerate(self.features) if not f.startswith("ctx_")]
        self.pipeline = pipeline
        self.name = name
        self.blend: dict = {"model_coef": 0.0, "market_coef": 1.0, "available": False}

    def _mirror(self, x):
        x = np.array(x, float, copy=True)
        x[:, self.diff_idx] *= -1
        return x

    def fit(self, X, y, template, sample_weight=None):
        self.pipeline = clone(template)
        kw = {} if sample_weight is None else {"model__sample_weight": np.concatenate([sample_weight, sample_weight])}
        self.pipeline.fit(np.vstack([X, self._mirror(X)]), np.concatenate([y, 1 - y]), **kw)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        both = self.pipeline.predict_proba(np.vstack([X, self._mirror(X)]))[:, 1]
        n = len(X)
        return np.clip(0.5 * (both[:n] + 1 - both[n:]), 0.01, 0.99)

    def predict_rows(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[r.get(f, np.nan) for f in self.features] for r in rows], float)
        return self.predict(X)


def _ll(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.log(p)))


def walk_forward(rows: pd.DataFrame, features: list[str], template, years: list[int]) -> pd.Series:
    """Probability that the actual WINNER wins, each year predicted from earlier years only."""
    X, y = rows[features].to_numpy(float), rows["_y"].to_numpy(int)
    out = []
    for year in years:
        tr = (rows["_year"] < year).to_numpy()
        te = (rows["_year"] == year).to_numpy()
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        m = MatchModel(features).fit(X[tr], y[tr], template, recency_weights(rows["_year"].to_numpy()[tr], year))
        p = m.predict(X[te])
        out.append(pd.Series(np.where(y[te] == 1, p, 1 - p), index=rows.index[te]))
        print(f"  walk-forward {year}: {te.sum()} matches", flush=True)
    return pd.concat(out) if out else pd.Series(dtype=float)


def fit_blend(p_model_w, p_market_w, seed=7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    side = rng.random(len(p_model_w)) < 0.5
    x = np.c_[logit(np.where(side, p_model_w, 1 - p_model_w)), logit(np.where(side, p_market_w, 1 - p_market_w))]
    yb = side.astype(int)
    m = LogisticRegression(C=1e6, fit_intercept=False, max_iter=2000).fit(x, yb)
    a, b = (float(v) for v in m.coef_[0])
    if a < 0:  # never bet against our own model; refit market-only
        b = float(LogisticRegression(C=1e6, fit_intercept=False, max_iter=2000).fit(x[:, 1:], yb).coef_[0][0])
        a = 0.0
    return a, b


def apply_blend(p_model, p_market, a: float, b: float) -> np.ndarray:
    return sigmoid(a * logit(p_model) + b * logit(p_market))


def walk_forward_blend(p_model_w: np.ndarray, p_market_w: np.ndarray, years: np.ndarray, min_train: int = 1500) -> np.ndarray:
    """Fair probability of the winner, each season blended with weights fitted on earlier
    seasons only. NaN for seasons without enough earlier data (the first one)."""
    out = np.full(len(years), np.nan)
    for year in sorted(set(years.tolist())):
        tr, te = years < year, years == year
        if tr.sum() < min_train:
            continue
        a, b = fit_blend(p_model_w[tr], p_market_w[tr])
        out[te] = apply_blend(p_model_w[te], p_market_w[te], a, b)
    return out


# ---------------------------------------------------------------------------
# Grand Slam over/under 3.5 sets
# ---------------------------------------------------------------------------
SETS_FEATURES = ["iid_logit", "serve_sum", "serve_gap", "speed_c", "long_hist", "slam_exp", "injury_sum", "round"]


def sets_features(fav_p, serve_a, serve_b, ret_a, ret_b, speed, long_a, long_b, n_a, n_b, inj_a, inj_b, rnd) -> np.ndarray:
    fav_p = np.asarray(fav_p, float)
    iid = np.array([over35_from_set_prob(set_prob_from_match(p, 5)) for p in fav_p])
    serve_a, serve_b, ret_a, ret_b = (np.asarray(v, float) for v in (serve_a, serve_b, ret_a, ret_b))
    return np.c_[
        logit(iid),
        serve_a + serve_b - 1.26,                       # strong servers -> tighter sets, more tiebreaks
        (serve_a - ret_b) - (serve_b - ret_a),          # serve/return mismatch
        np.asarray(speed, float) - 1.0,                 # court speed index
        (np.asarray(long_a, float) + np.asarray(long_b, float)) / 2 - 0.55,  # players' history of long Slam matches
        np.log1p(np.minimum(n_a, n_b)),                 # Slam best-of-5 experience
        np.asarray(inj_a, float) + np.asarray(inj_b, float),
        np.asarray(rnd, float),
    ]


def sets_rows(rows: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    g = rows[(rows["_level"] == "G") & (rows["_best_of"] == 5) & (rows["_sets"] >= 3)].copy()
    if g.empty:
        return g
    g = attach(g, odds)
    pe = sigmoid(g["elo_prob_logit"].to_numpy())
    mk = no_vig_array(g["ps_w"], g["ps_l"])
    hasm = np.isfinite(mk)
    g["fav_p"] = np.where(hasm, np.maximum(mk, 1 - mk), np.maximum(pe, 1 - pe))
    g["p_source"] = np.where(hasm, "pinnacle", "elo")
    g["over"] = (g["_sets"] >= 4).astype(int)
    return g


def _sets_X(g: pd.DataFrame) -> np.ndarray:
    return sets_features(g["fav_p"], g["_serve_a"], g["_serve_b"], g["_ret_a"], g["_ret_b"], g["ctx_speed"],
                         g["_slam_long_a"], g["_slam_long_b"], g["_slam_n_a"], g["_slam_n_b"], g["_inj_a"], g["_inj_b"], g["ctx_round"])


def train_sets(rows: pd.DataFrame, odds: pd.DataFrame) -> dict:
    g = sets_rows(rows, odds)
    if len(g) < 1000:
        return {"available": False, "reason": "not enough Grand Slam matches"}
    X, y = _sets_X(g), g["over"].to_numpy()
    years = sorted(g["_year"].unique())
    eval_years = [yr for yr in years if yr >= max(years) - 9]
    preds, base, iid = [], [], []
    for yr in eval_years:
        tr, te = (g["_year"] < yr).to_numpy(), (g["_year"] == yr).to_numpy()
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = LogisticRegression(C=1.0, max_iter=4000).fit(X[tr], y[tr])
        preds.append(pd.Series(m.predict_proba(X[te])[:, 1], index=g.index[te]))
        base.append(pd.Series(y[tr].mean(), index=g.index[te]))
    p, b = pd.concat(preds), pd.concat(base)
    e = g.loc[p.index]
    ll = lambda yy, pp: float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1]))  # noqa: E731
    iid_p = sigmoid(X[g.index.get_indexer(p.index), 0])
    calib = (pd.DataFrame({"p": p.values, "y": e["over"].values}).assign(bin=lambda d: pd.cut(d.p, [0, .4, .5, .6, .7, 1]))
             .groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index())
    calib["bin"] = calib["bin"].astype(str)
    final = LogisticRegression(C=1.0, max_iter=4000).fit(X, y)
    joblib.dump({"model": final, "features": SETS_FEATURES}, SETS_MODEL, compress=3)
    return {
        "available": True,
        "matches": int(len(g)), "over_rate": float(y.mean()),
        "eval_years": [int(v) for v in eval_years], "eval_matches": int(len(e)),
        "log_loss": ll(e["over"], p.values), "base_rate_log_loss": ll(e["over"], b.values),
        "independent_sets_log_loss": ll(e["over"], iid_p),
        "brier": float(brier_score_loss(e["over"], p.values)),
        "calibration": calib.to_dict("records"),
        "coefficients": dict(zip(SETS_FEATURES, [float(c) for c in final.coef_[0]])),
        "note": ("Probability source: Pinnacle moneyline when available (else rating model) -> per-set win "
                 "probability -> P(4+ sets) -> logistic correction. No historical total-sets prices exist in the "
                 "archive, so betting ROI on this market cannot be backtested; track it live."),
    }


def predict_over35(fav_p, a_state: dict, b_state: dict, speed: float, round_idx: float) -> float | None:
    if not SETS_MODEL.exists():
        return None
    bundle = joblib.load(SETS_MODEL)
    X = sets_features([fav_p], [a_state["serve"]], [b_state["serve"]], [a_state["ret"]], [b_state["ret"]], [speed],
                      [a_state["slam_long"]], [b_state["slam_long"]], [a_state["slam_n"]], [b_state["slam_n"]],
                      [a_state["inj_recent"]], [b_state["inj_recent"]], [round_idx])
    return float(bundle["model"].predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
DEFAULT_MIN_EV = 0.03      # the price must beat the sharp no-vig price by this much
DEFAULT_MAX_ODDS = 8.0     # no bets on bigger outsiders


def _scoreboard(t: pd.DataFrame, col: str) -> dict:
    return {"log_loss": _ll(t[col]), "accuracy": float((t[col] > 0.5).mean())}


def strategy_backtests(t: pd.DataFrame) -> dict:
    """Replay betting rules on the seasons where the blended fair price is out-of-sample.

    ``t`` needs p_winner (model), p_market (Pinnacle no-vig) and p_fair (walk-forward blend),
    all for the eventual winner, plus closing prices: Pinnacle, the books' average and the
    best price on offer. Every rule is replayed on exactly the same matches.
    """
    ok = t["p_fair"].notna() & t["ps_w"].gt(1) & t["ps_l"].gt(1) & t["avg_w"].gt(1) & t["avg_l"].gt(1) & t["max_w"].gt(1) & t["max_l"].gt(1)
    t = t[ok].sort_values("_date")
    if len(t) < 500:
        return {}
    prices = {"pinnacle": (t["ps_w"], t["ps_l"]), "best": (t["max_w"], t["max_l"]),
              "halfway": ((t["avg_w"] + t["max_w"]) / 2, (t["avg_l"] + t["max_l"]) / 2)}
    rules = {
        # the rule used before v2: the model's probability is the fair price
        "old_model_only": dict(p=t["p_winner"], min_ev=0.05, min_edge=0.05),
        # v2 (default 3%): the price must beat the sharp price with its margin removed by min_ev,
        # the blended fair price must not disagree, and the stake is sized on the blended price
        "new_ev2": dict(p=t["p_market"], min_ev=0.02, max_odds=DEFAULT_MAX_ODDS, stake_p_winner=t["p_fair"]),
        "new_ev3": dict(p=t["p_market"], min_ev=0.03, max_odds=DEFAULT_MAX_ODDS, stake_p_winner=t["p_fair"]),
        "new_ev5": dict(p=t["p_market"], min_ev=0.05, max_odds=DEFAULT_MAX_ODDS, stake_p_winner=t["p_fair"]),
        # the same rule with no model at all: how much comes from price shopping alone
        "price_only_ev3": dict(p=t["p_market"], min_ev=0.03, max_odds=DEFAULT_MAX_ODDS),
    }
    out = {"matches": int(len(t)), "years": [int(t["_year"].min()), int(t["_year"].max())], "rules": {}, "by_period": []}
    for rule, kw in rules.items():
        kw = dict(kw)
        p = kw.pop("p")
        out["rules"][rule] = {price: backtest(p, ow, ol, market_w=t["p_market"], **kw) for price, (ow, ol) in prices.items()}
    # The default rule season by season, at the best price: is the edge still there?
    for yr in sorted(t["_year"].unique()):
        g = t[t["_year"] == yr]
        r = backtest(g["p_market"], g["max_w"], g["max_l"], min_ev=DEFAULT_MIN_EV, market_w=g["p_market"],
                     max_odds=DEFAULT_MAX_ODDS, stake_p_winner=g["p_fair"])
        out["by_period"].append({"year": int(yr), "matches": int(len(g)), **r})
    return out


def train_all(rows: pd.DataFrame, engine, data_report: dict | None = None) -> dict:
    features = [c for c in rows.columns if not c.startswith("_")]
    X, y = rows[features].to_numpy(float), rows["_y"].to_numpy(int)
    years_all = rows["_year"].to_numpy()
    latest = int(rows["_year"].max())
    holdout = latest - 1 if (rows["_year"] == latest).sum() < 2000 else latest
    sel_years = [holdout - 2, holdout - 1, holdout]

    print("model selection", sel_years, flush=True)
    selection = {}
    for name, template in CANDIDATES.items():
        lls = []
        for yr in sel_years:
            tr, te = (rows["_year"] < yr).to_numpy(), (rows["_year"] == yr).to_numpy()
            p = MatchModel(features).fit(X[tr], y[tr], template, recency_weights(years_all[tr], yr)).predict(X[te])
            lls.append(float(log_loss(y[te], p, labels=[0, 1])))
        selection[name] = float(np.mean(lls))
        print(f"  {name}: mean log loss {selection[name]:.4f}", flush=True)
    best = min(selection, key=selection.get)

    print("walk-forward evaluation", flush=True)
    oof = walk_forward(rows, features, CANDIDATES[best], list(range(2018, latest + 1)))
    t = attach(rows.loc[oof.index].assign(p_winner=oof.values), load_odds())
    t["p_pin"] = no_vig_array(t["ps_w"], t["ps_l"])        # sharp reference
    t["p_avg"] = no_vig_array(t["avg_w"], t["avg_l"])      # the books' consensus
    pin = t[t["p_pin"].notna()].copy()
    pin["p_market"] = pin["p_pin"]
    avg = t[t["p_pin"].isna() & t["p_avg"].notna()].copy()
    avg["p_market"] = avg["p_avg"]

    per_year = []
    for yr in sorted(t["_year"].unique()):
        gp, ga = pin[pin["_year"] == yr], avg[avg["_year"] == yr]
        g, src = (gp, "Pinnacle close") if len(gp) >= 200 else (ga, "market-average close")
        if len(g) < 100:
            continue
        per_year.append({"year": int(yr), "matches": int(len(g)), "market_source": src,
                         "model_log_loss": _ll(g["p_winner"]), "market_log_loss": _ll(g["p_market"]),
                         "model_accuracy": float((g["p_winner"] > 0.5).mean()), "market_accuracy": float((g["p_market"] > 0.5).mean())})

    # ---- market blend: fair price = a*logit(model) + b*logit(market)
    a, b = fit_blend(pin["p_winner"].to_numpy(), pin["p_market"].to_numpy())
    cons = t[t["p_avg"].notna()]
    ac, bc = fit_blend(cons["p_winner"].to_numpy(), cons["p_avg"].to_numpy())
    pin["p_fair"] = walk_forward_blend(pin["p_winner"].to_numpy(), pin["p_market"].to_numpy(), pin["_year"].to_numpy())
    bl = pin[pin["p_fair"].notna()]
    blend_eval = {"matches": int(len(bl)), "years": [int(bl["_year"].min()), int(bl["_year"].max())] if len(bl) else [],
                  "fair_log_loss": _ll(bl["p_fair"]) if len(bl) else None,
                  "market_log_loss": _ll(bl["p_market"]) if len(bl) else None,
                  "model_log_loss": _ll(bl["p_winner"]) if len(bl) else None}
    backtests = strategy_backtests(pin)

    print("final fit on all rows", flush=True)
    model = MatchModel(features, name=best).fit(X, y, CANDIDATES[best], recency_weights(years_all, latest))
    model.blend = {"available": True, "model_coef": a, "market_coef": b, "fitted_matches": int(len(pin)),
                   "consensus_model_coef": ac, "consensus_market_coef": bc, "consensus_fitted_matches": int(len(cons))}
    joblib.dump(model, MATCH_MODEL, compress=3)

    print("Grand Slam sets model", flush=True)
    sets_metrics = train_sets(rows, load_odds())

    joblib.dump(engine, ENGINE_PATH, compress=3)

    prop = (1 / pin["ps_w"]) / (1 / pin["ps_w"] + 1 / pin["ps_l"])
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_match_date": str(engine.last_date),
        "latest_priced_match": str(getattr(engine, "last_priced_date", None)),
        "training_matches": int(len(rows)),
        "features": features,
        "selected_model": best,
        "selection_log_loss": selection,
        "recency_half_life_years": RECENCY_HALF_LIFE_YEARS,
        "market_benchmark": {
            "years": [int(v) for v in sorted(pin["_year"].unique())],
            "matches": int(len(pin)),
            "model_log_loss": _ll(pin["p_winner"]), "pinnacle_log_loss": _ll(pin["p_market"]),
            "pinnacle_log_loss_simple_devig": _ll(prop),
            "model_accuracy": float((pin["p_winner"] > 0.5).mean()), "pinnacle_accuracy": float((pin["p_market"] > 0.5).mean()),
            "per_year": per_year,
        },
        "market_blend": model.blend,
        "blend_benchmark": blend_eval,
        "backtests": backtests,
        "sets_model": sets_metrics,
        "data": data_report or {},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
