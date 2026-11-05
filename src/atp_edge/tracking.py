"""Automatic record of every prediction and every recommended bet.

data/predictions/predictions_log.csv  one row per match: model / fair / market
                                       probabilities (latest before start) and result
data/predictions/bets_log.csv          one row per recommended bet: price when first
                                       flagged, closing Pinnacle price, result, P/L, CLV

Both files are written by GitHub Actions and committed, so the history is public,
timestamped and cannot be edited after the fact without it showing in git.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .betting import no_vig
from .engine import score_info
from .config import PREDICTIONS
from .predict import pair_key, same_player
from .tournaments import normalize

PRED_LOG = PREDICTIONS / "predictions_log.csv"
BETS_LOG = PREDICTIONS / "bets_log.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def match_id(m: dict) -> str:
    day = datetime.fromtimestamp(m["start_ts"], tz=timezone.utc).strftime("%Y-%m-%d") if m.get("start_ts") else "na"
    a, b = pair_key(m["player_a"], m["player_b"])
    return f"{day}|{a}|{b}"


def record(slate: dict) -> dict:
    """Upsert predictions and bets from a slate. Only pre-match snapshots are written."""
    now = datetime.now(timezone.utc).timestamp()
    preds, bets = _read(PRED_LOG), _read(BETS_LOG)
    pred_rows, bet_rows = [], []
    for m in slate.get("matches", []):
        if m.get("start_ts") and m["start_ts"] < now:
            continue
        mid = match_id(m)
        sets = m.get("sets") or {}
        pred_rows.append({
            "match_id": mid, "updated_at": _now(), "start_ts": m.get("start_ts"), "tournament": m.get("event"),
            "level": m.get("level"), "surface": m.get("surface"), "round": m.get("round"),
            "player_a": m["player_a"], "player_b": m["player_b"],
            "p_model_a": m.get("p_model_a"), "p_fair_a": m.get("p_fair_a"), "p_market_a": m.get("p_market_a"),
            "pinnacle_a": m.get("pinnacle_a"), "pinnacle_b": m.get("pinnacle_b"),
            "p_over35": sets.get("p_over35"), "pinnacle_over35": sets.get("pinnacle_over"), "pinnacle_under35": sets.get("pinnacle_under"),
            "court_speed": m.get("court_speed"),
        })
        for rec in (m.get("recommendation"), sets.get("recommendation")):
            if not rec:
                continue
            bid = f"{mid}|{rec['market']}|{normalize(rec['selection'])}"
            bet_rows.append({
                "bet_id": bid, "match_id": mid, "flagged_at": _now(), "start_ts": m.get("start_ts"),
                "tournament": m.get("event"), "level": m.get("level"), "player_a": m["player_a"], "player_b": m["player_b"],
                "market": rec["market"], "selection": rec["selection"], "price": rec["price"], "book": rec.get("book"),
                "probability": rec["probability"], "ev": rec["ev"], "stake": rec["stake"], "strategy": m.get("strategy"),
            })
    if pred_rows:
        new = pd.DataFrame(pred_rows)
        # Opening snapshot (first time we saw the match) is kept forever; the latest
        # pre-match snapshot keeps overwriting pinnacle_a/b and becomes the close.
        opens = preds.set_index("match_id") if not preds.empty and "open_pinnacle_a" in preds.columns else None
        firsts = []
        for r in pred_rows:
            prev = opens.loc[r["match_id"]] if opens is not None and r["match_id"] in opens.index else None
            if isinstance(prev, pd.DataFrame):
                prev = prev.iloc[0]
            keep_prev = prev is not None and pd.notna(prev.get("open_pinnacle_a"))
            firsts.append({
                "first_seen": prev.get("first_seen") if prev is not None and pd.notna(prev.get("first_seen")) else r["updated_at"],
                "open_pinnacle_a": prev.get("open_pinnacle_a") if keep_prev else r["pinnacle_a"],
                "open_pinnacle_b": prev.get("open_pinnacle_b") if keep_prev else r["pinnacle_b"],
                "open_p_model_a": prev.get("open_p_model_a") if keep_prev else r["p_model_a"],
                "snapshots": (int(prev.get("snapshots") or 0) + 1) if prev is not None else 1,
            })
        new = pd.concat([new, pd.DataFrame(firsts)], axis=1)
        if not preds.empty:
            keep = preds[~preds["match_id"].isin(new["match_id"]) | preds.get("winner", pd.Series(index=preds.index, dtype=object)).notna()]
            # keep settled rows untouched; refresh open ones with the latest pre-match snapshot
            new = new[~new["match_id"].isin(keep["match_id"])]
            preds = pd.concat([keep, new], ignore_index=True)
        else:
            preds = new
        preds = add_lean_clv(preds)
        preds.to_csv(PRED_LOG, index=False)
    added = 0
    if bet_rows:
        new = pd.DataFrame(bet_rows)
        if not bets.empty:
            new = new[~new["bet_id"].isin(bets["bet_id"])]  # first flag (price taken) is never overwritten
        added = len(new)
        bets = pd.concat([bets, new], ignore_index=True) if not bets.empty else new
        bets.to_csv(BETS_LOG, index=False)
    return {"predictions": len(pred_rows), "new_bets": added}


def settle(matches: pd.DataFrame) -> dict:
    """Attach results from the match database (and closing Pinnacle prices) to the logs."""
    preds, bets = _read(PRED_LOG), _read(BETS_LOG)
    if preds.empty:
        return {"settled_predictions": 0, "settled_bets": 0}
    m = matches.copy()
    m["mdate"] = pd.to_datetime(pd.to_numeric(m["tourney_date"], errors="coerce").astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    m = m[m["mdate"] >= pd.Timestamp.now() - pd.Timedelta(days=60)]
    m["pkey"] = [pair_key(w, l) for w, l in zip(m["winner_name"], m["loser_name"])]
    by_pair = {}
    for r in m.itertuples(index=False):
        by_pair.setdefault(r.pkey, []).append(r)
    for c in ("winner", "sets", "score"):
        if c not in preds.columns:
            preds[c] = np.nan
    preds["winner"] = preds["winner"].astype(object)   # text columns must not be float-typed
    preds["score"] = preds["score"].astype(object)
    settled = 0
    for i, r in preds[preds["winner"].isna()].iterrows():
        start = pd.to_datetime(r["start_ts"], unit="s") if pd.notna(r["start_ts"]) else None
        for cand in by_pair.get(pair_key(r["player_a"], r["player_b"]), []):
            if start is not None and not (-16 <= (start.normalize() - cand.mdate).days <= 3):
                continue
            info = score_info(cand.score)
            preds.at[i, "winner"] = r["player_a"] if same_player(cand.winner_name, r["player_a"]) else r["player_b"]
            preds.at[i, "sets"] = info["sets"] if not info["incomplete"] else np.nan
            preds.at[i, "score"] = cand.score
            settled += 1
            break
    preds = add_lean_clv(preds)
    preds.to_csv(PRED_LOG, index=False)

    nb = 0
    if not bets.empty:
        for c in ("result", "pl_units", "closing_price", "clv"):
            if c not in bets.columns:
                bets[c] = np.nan
        bets["result"] = bets["result"].astype(object)
        lookup = preds.set_index("match_id")
        for i, b in bets[bets["result"].isna()].iterrows():
            if b["match_id"] not in lookup.index:
                continue
            p = lookup.loc[b["match_id"]]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[-1]
            if pd.isna(p.get("winner")):
                continue
            if b["market"] == "Moneyline":
                won = same_player(b["selection"], p["winner"])
                a_side = same_player(b["selection"], p["player_a"])
                close_odds = p["pinnacle_a"] if a_side else p["pinnacle_b"]
                close_fair = no_vig(p["pinnacle_a"], p["pinnacle_b"])
                close_fair = close_fair if a_side else (1 - close_fair if close_fair else None)
            else:
                if pd.isna(p.get("sets")):
                    bets.at[i, "result"] = "void"
                    bets.at[i, "pl_units"] = 0.0
                    nb += 1
                    continue
                over = float(p["sets"]) >= 4
                won = over if b["selection"].startswith("Over") else not over
                is_over = b["selection"].startswith("Over")
                close_odds = p.get("pinnacle_over35") if is_over else p.get("pinnacle_under35")
                cf = no_vig(p.get("pinnacle_over35"), p.get("pinnacle_under35"))
                close_fair = (cf if is_over else (1 - cf)) if cf else None
            bets.at[i, "result"] = "won" if won else "lost"
            bets.at[i, "pl_units"] = float(b["stake"]) * ((float(b["price"]) - 1) if won else -1.0)
            bets.at[i, "closing_price"] = close_odds
            bets.at[i, "clv"] = (float(b["price"]) * close_fair - 1) if close_fair else np.nan
            nb += 1
        bets.to_csv(BETS_LOG, index=False)
    return {"settled_predictions": settled, "settled_bets": nb}


def add_lean_clv(preds: pd.DataFrame) -> pd.DataFrame:
    """CLV of the model's own lean on every match - the cleanest test of skill.

    Lean = the side where the model's probability (when first seen) exceeded the
    market's. CLV = opening Pinnacle price x closing Pinnacle no-vig probability - 1.
    Positive on average = the model spots prices that later move its way.
    """
    if preds.empty or "open_pinnacle_a" not in preds.columns:
        return preds
    leans, clvs, moves = [], [], []
    for r in preds.itertuples(index=False):
        oa, ob = getattr(r, "open_pinnacle_a", None), getattr(r, "open_pinnacle_b", None)
        ca, cb = getattr(r, "pinnacle_a", None), getattr(r, "pinnacle_b", None)
        p0 = no_vig(oa, ob)
        pc = no_vig(ca, cb)
        pm = getattr(r, "open_p_model_a", None)
        try:
            pm = float(pm)
        except Exception:
            pm = None
        if p0 is None or pc is None or pm is None or not np.isfinite(pm):
            leans.append(np.nan); clvs.append(np.nan); moves.append(np.nan)
            continue
        if pm >= p0:
            leans.append(r.player_a); clvs.append(float(oa) * pc - 1)
        else:
            leans.append(r.player_b); clvs.append(float(ob) * (1 - pc) - 1)
        moves.append(pc - p0)
    out = preds.copy()
    out["model_lean"], out["lean_clv"], out["market_move_a"] = leans, clvs, moves
    return out


def summary() -> dict:
    preds, bets = _read(PRED_LOG), _read(BETS_LOG)
    out = {"predictions": int(len(preds)), "bets": int(len(bets))}
    if not preds.empty and "winner" in preds.columns:
        s = preds.dropna(subset=["winner"])
        if len(s):
            y = np.array([same_player(w, a) for w, a in zip(s["winner"], s["player_a"])], float)
            def ll(p):
                p = np.clip(pd.to_numeric(p, errors="coerce").to_numpy(float), 1e-6, 1 - 1e-6)
                ok = np.isfinite(p)
                return float(-np.mean(y[ok] * np.log(p[ok]) + (1 - y[ok]) * np.log(1 - p[ok]))) if ok.any() else None
            out.update({"settled": int(len(s)), "model_log_loss": ll(s["p_model_a"]), "fair_log_loss": ll(s["p_fair_a"]),
                        "market_log_loss": ll(s["p_market_a"]),
                        "model_accuracy": float(((pd.to_numeric(s["p_model_a"]) > 0.5) == (y == 1)).mean())})
    if not bets.empty and "result" in bets.columns:
        done = bets[bets["result"].isin(["won", "lost"])]
        if len(done):
            out.update({"settled_bets": int(len(done)), "units_staked": float(done["stake"].sum()),
                        "units_pl": float(done["pl_units"].sum()),
                        "roi": float(done["pl_units"].sum() / done["stake"].sum()) if done["stake"].sum() else None,
                        "avg_clv": float(pd.to_numeric(done["clv"], errors="coerce").mean()),
                        "beat_close_rate": float((pd.to_numeric(done["clv"], errors="coerce") > 0).mean())})
    if not preds.empty and "lean_clv" in preds.columns:
        started = pd.to_numeric(preds["start_ts"], errors="coerce") < datetime.now(timezone.utc).timestamp()
        lc = pd.to_numeric(preds.loc[started, "lean_clv"], errors="coerce").dropna()
        if len(lc):
            out.update({"lean_clv_n": int(len(lc)), "lean_clv": float(lc.mean()),
                        "lean_clv_se": float(lc.std(ddof=1) / np.sqrt(len(lc))) if len(lc) > 1 else None,
                        "lean_beat_close_rate": float((lc > 0).mean())})
    return out
