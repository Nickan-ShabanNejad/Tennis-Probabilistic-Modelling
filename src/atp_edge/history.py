"""Replay the match history through the engine to create leakage-free training rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Engine, parse_date, score_info
from .tournaments import BETTABLE_LEVELS


def replay(matches: pd.DataFrame, engine: Engine, seed: int = 7, progress: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    records = matches.to_dict("records")
    for i, r in enumerate(records):
        if progress and i % 20000 == 0:
            print(f"  replay {i:,}/{len(records):,}", flush=True)
        surface = str(r.get("surface", "")).title()
        level = str(r.get("tourney_level", ""))
        info = score_info(r.get("score"))
        usable = (not info["incomplete"]) and surface in {"Hard", "Clay", "Grass"} and level in BETTABLE_LEVELS
        if usable:
            flip = bool(rng.random() < 0.5)
            a, b = (str(r["loser_id"]), str(r["winner_id"])) if flip else (str(r["winner_id"]), str(r["loser_id"]))
            best_of = int(r["best_of"]) if r.get("best_of") in (3, 5) else (5 if level == "G" else 3)
            ctx = {"tournament": r.get("tourney_name", ""), "surface": surface, "level": level, "best_of": best_of,
                   "indoor": str(r.get("indoor", "")).upper() in {"I", "1", "1.0", "TRUE"}, "round": r.get("round", ""),
                   "date": r["tourney_date"]}
            out = engine.features(a, b, ctx)
            row = dict(out["x"])
            row.update({
                "_y": 0 if flip else 1, "_date": parse_date(r["tourney_date"]), "_year": parse_date(r["tourney_date"]).year,
                "_a": r["loser_name"] if flip else r["winner_name"], "_b": r["winner_name"] if flip else r["loser_name"],
                "_winner": r["winner_name"], "_loser": r["loser_name"], "_tournament": r.get("tourney_name", ""),
                "_level": level, "_best_of": best_of, "_sets": info["sets"], "_surface": surface,
                "_slam_straight_a": out["meta"]["a_state"]["slam_straight"], "_slam_straight_b": out["meta"]["b_state"]["slam_straight"],
                "_slam_long_a": out["meta"]["a_state"]["slam_long"], "_slam_long_b": out["meta"]["b_state"]["slam_long"],
                "_slam_n_a": out["meta"]["a_state"]["slam_n"], "_slam_n_b": out["meta"]["b_state"]["slam_n"],
                "_serve_a": out["meta"]["a_state"]["serve"], "_serve_b": out["meta"]["b_state"]["serve"],
                "_ret_a": out["meta"]["a_state"]["ret"], "_ret_b": out["meta"]["b_state"]["ret"],
                "_inj_a": out["meta"]["a_state"]["inj_recent"], "_inj_b": out["meta"]["b_state"]["inj_recent"],
                "_speed_source": out["meta"]["speed"]["source"],
            })
            rows.append(row)
        engine.update(r)
    return pd.DataFrame(rows)
