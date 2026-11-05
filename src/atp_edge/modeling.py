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

from .betting import backtest, logit, over35_from_set_prob, set_prob_from_match, sigmoid
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

    def fit(self, X, y, template):
        self.pipeline = clone(template)
        self.pipeline.fit(np.vstack([X, self._mirror(X)]), np.concatenate([y, 1 - y]))
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
        m = MatchModel(features).fit(X[tr], y[tr], template)
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
    hasm = (g["ps_w"].gt(1) & g["ps_l"].gt(1)).to_numpy()
    mk = (1 / g["ps_w"] / (1 / g["ps_w"] + 1 / g["ps_l"])).to_numpy()
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
def train_all(rows: pd.DataFrame, engine, data_report: dict | None = None) -> dict:
    features = [c for c in rows.columns if not c.startswith("_")]
    X, y = rows[features].to_numpy(float), rows["_y"].to_numpy(int)
    latest = int(rows["_year"].max())
    holdout = latest - 1 if (rows["_year"] == latest).sum() < 2000 else latest
    sel_years = [holdout - 2, holdout - 1, holdout]

    print("model selection", sel_years, flush=True)
    selection = {}
    for name, template in CANDIDATES.items():
        lls = []
        for yr in sel_years:
            tr, te = (rows["_year"] < yr).to_numpy(), (rows["_year"] == yr).to_numpy()
            p = MatchModel(features).fit(X[tr], y[tr], template).predict(X[te])
            lls.append(float(log_loss(y[te], p, labels=[0, 1])))
        selection[name] = float(np.mean(lls))
        print(f"  {name}: mean log loss {selection[name]:.4f}", flush=True)
    best = min(selection, key=selection.get)

    print("walk-forward evaluation", flush=True)
    oof = walk_forward(rows, features, CANDIDATES[best], list(range(2018, latest + 1)))
    t = attach(rows.loc[oof.index].assign(p_winner=oof.values), load_odds())
    pin = t[t["ps_w"].gt(1) & t["ps_l"].gt(1)].copy()
    pin["p_market"] = (1 / pin["ps_w"]) / (1 / pin["ps_w"] + 1 / pin["ps_l"])
    avg = t[~(t["ps_w"].gt(1) & t["ps_l"].gt(1)) & t["avg_w"].gt(1) & t["avg_l"].gt(1)].copy()
    avg["p_market"] = (1 / avg["avg_w"]) / (1 / avg["avg_w"] + 1 / avg["avg_l"])

    per_year = []
    for yr in sorted(t["_year"].unique()):
        gp, ga = pin[pin["_year"] == yr], avg[avg["_year"] == yr]
        g, src = (gp, "Pinnacle close") if len(gp) >= 200 else (ga, "market-average close")
        if len(g) < 100:
            continue
        per_year.append({"year": int(yr), "matches": int(len(g)), "market_source": src,
                         "model_log_loss": _ll(g["p_winner"]), "market_log_loss": _ll(g["p_market"]),
                         "model_accuracy": float((g["p_winner"] > 0.5).mean()), "market_accuracy": float((g["p_market"] > 0.5).mean())})

    a, b = fit_blend(pin["p_winner"].to_numpy(), pin["p_market"].to_numpy())
    blended = sigmoid(a * logit(pin["p_winner"]) + b * logit(pin["p_market"]))
    backtests = {
        "model_only_ev2_edge2": backtest(pin["p_winner"], pin["ps_w"], pin["ps_l"], min_ev=0.02, min_edge=0.02, market_w=pin["p_market"]),
        "model_only_ev5_edge5": backtest(pin["p_winner"], pin["ps_w"], pin["ps_l"], min_ev=0.05, min_edge=0.05, market_w=pin["p_market"]),
        "market_anchored_ev2": backtest(blended, pin["ps_w"], pin["ps_l"], min_ev=0.02, market_w=pin["p_market"]),
    }

    print("final fit on all rows", flush=True)
    model = MatchModel(features, name=best).fit(X, y, CANDIDATES[best])
    model.blend = {"available": True, "model_coef": a, "market_coef": b, "fitted_matches": int(len(pin))}
    joblib.dump(model, MATCH_MODEL, compress=3)

    print("Grand Slam sets model", flush=True)
    sets_metrics = train_sets(rows, load_odds())

    joblib.dump(engine, ENGINE_PATH, compress=3)

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_match_date": str(engine.last_date),
        "training_matches": int(len(rows)),
        "features": features,
        "selected_model": best,
        "selection_log_loss": selection,
        "market_benchmark": {
            "years": [int(v) for v in sorted(pin["_year"].unique())],
            "matches": int(len(pin)),
            "model_log_loss": _ll(pin["p_winner"]), "pinnacle_log_loss": _ll(pin["p_market"]),
            "model_accuracy": float((pin["p_winner"] > 0.5).mean()), "pinnacle_accuracy": float((pin["p_market"] > 0.5).mean()),
            "per_year": per_year,
        },
        "market_blend": model.blend,
        "backtests": backtests,
        "sets_model": sets_metrics,
        "data": data_report or {},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
