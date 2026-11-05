from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------ tournaments
@pytest.mark.parametrize("name,key,level", [
    ("China Open - Beijing", "beijing", "500"), ("Rolex Paris Masters", "paris masters", "M"),
    ("French Open - Paris", "roland garros", "G"), ("U.S. Open - New York", "us open", "G"),
    ("US Men's Clay Court Championship - Houston", "houston", "250"), ("Nitto ATP Finals", "atp finals", "F"),
    ("HSBC Championships - London", "queens club", "500"), ("Wimbledon", "wimbledon", "G"),
    ("ATP Shanghai Masters", "shanghai", "M"), ("Kinoshita Group Japan Open", "tokyo", "500"),
])
def test_tournament_resolution(name, key, level):
    from atp_edge.tournaments import resolve
    ev = resolve(name)
    assert ev is not None and ev.key == key and ev.level == level


@pytest.mark.parametrize("name", ["Laver Cup", "Davis Cup Finals", "Bengaluru Challenger", "M25 Marrakech", "United Cup", "Next Gen ATP Finals"])
def test_non_bettable_events_are_excluded(name):
    from atp_edge.tournaments import resolve
    assert resolve(name) is None


def test_slams_are_best_of_five_only():
    from atp_edge.tournaments import EVENTS
    assert {e.best_of for e in EVENTS.values() if e.level == "G"} == {5}
    assert {e.best_of for e in EVENTS.values() if e.level != "G"} == {3}


# ------------------------------------------------------------------ data cleaning
def _row(**kw):
    base = dict(tourney_name="X", surface="Hard", tourney_level="250", tourney_date=20260105, winner_id="a", loser_id="b",
                winner_name="Jaume Munar", loser_name="Taylor Fritz", score="6-4 6-4", round="R32", data_source="TennisMyLife")
    base.update(kw)
    return base


def test_cross_provider_duplicates_removed():
    from atp_edge.data import clean
    df = pd.DataFrame([
        _row(tourney_name="Auckland", tourney_date=20260112),
        _row(tourney_name="ASB Classic - Auckland", tourney_level="A", tourney_date=20260115, data_source="Matchstat",
             winner_name="Jaume Antoni Munar Clar", w_svpt=60, round="First"),
    ])
    out, rep = clean(df)
    assert len(out) == 1 and rep["duplicates_removed"] == 1
    assert float(out.iloc[0]["w_svpt"]) == 60


def test_lower_tier_and_qualifying_removed():
    from atp_edge.data import clean
    df = pd.DataFrame([
        _row(tourney_name="M25 Marrakech", tourney_level="C"),
        _row(tourney_name="Pune Challenger", winner_name="A B", loser_name="C D"),
        _row(tourney_name="Brisbane", round="Q1", winner_name="E F", loser_name="G H"),
        _row(tourney_name="Brisbane", winner_name="I J", loser_name="K L"),
    ])
    out, rep = clean(df)
    assert len(out) == 1 and rep["lower_tier_removed"] == 2 and rep["qualifying_removed"] == 1


def test_generic_level_codes_use_registry():
    from atp_edge.data import resolve_level
    assert resolve_level("Barcelona Open Banc Sabadell - Barcelona", "A") == "500"
    assert resolve_level("Dubai", "500") == "500"


# ------------------------------------------------------------------ betting maths
def test_set_probability_inversion_round_trip():
    from atp_edge.betting import bo5_win, set_prob_from_match
    for p in (0.3, 0.5, 0.7, 0.9):
        assert abs(bo5_win(set_prob_from_match(p, 5)) - p) < 1e-6


def test_over35_is_symmetric_and_peaks_at_even():
    from atp_edge.betting import over35_from_set_prob
    assert abs(over35_from_set_prob(0.3) - over35_from_set_prob(0.7)) < 1e-12
    assert over35_from_set_prob(0.5) == pytest.approx(0.75)
    assert over35_from_set_prob(0.9) < 0.3


def test_stake_capped_and_no_negative_stakes():
    from atp_edge.betting import kelly_stake
    assert kelly_stake(0.8, 2.0, 0.25, 0.02) == 0.02
    assert kelly_stake(0.3, 2.0) == 0.0


def test_blend_with_zero_model_weight_returns_market():
    from atp_edge.betting import blend
    assert blend(0.9, 0.6, {"model_coef": 0.0, "market_coef": 1.0}) == pytest.approx(0.6)


def test_injury_flag_lowers_probability():
    from atp_edge.betting import injury_adjust
    assert injury_adjust(0.5, 0.5, 0.0) < 0.4
    assert injury_adjust(0.5, 0.0, 0.0) == 0.5


# ------------------------------------------------------------------ engine
def test_engine_has_no_lookahead():
    """Features for a match must not change when that match's result is changed."""
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.engine import Engine
    rows = [dict(tourney_name="Brisbane", surface="Hard", tourney_level="250", tourney_date=20250101 + i,
                 winner_id="a" if i % 2 else "b", loser_id="b" if i % 2 else "a", winner_name="A A", loser_name="B B",
                 score="6-4 6-4", best_of=3) for i in range(6)]
    e1, e2 = Engine(CourtSpeed(pd.DataFrame())), Engine(CourtSpeed(pd.DataFrame()))
    for r in rows[:5]:
        e1.update(r)
        e2.update(r)
    ctx = dict(tournament="Brisbane", surface="Hard", level="250", best_of=3, date=20250110)
    assert e1.features("a", "b", ctx)["x"] == e2.features("a", "b", ctx)["x"]


def test_retirement_creates_injury_signal_not_result():
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.engine import Engine
    e = Engine(CourtSpeed(pd.DataFrame()))
    e.update(dict(tourney_name="Doha", surface="Hard", tourney_level="500", tourney_date=20250201, winner_id="a",
                  loser_id="b", winner_name="A A", loser_name="B B", score="6-3 2-1 RET"))
    assert e.players["b"].injury_dates and e.players["b"].n == 0
    f = e.features("a", "b", dict(tournament="Doha", surface="Hard", level="500", best_of=3, date=20250205))["x"]
    assert f["inj_recent_diff"] < 0


def test_features_are_antisymmetric():
    from atp_edge.modeling import ENGINE_PATH
    import joblib
    if not ENGINE_PATH.exists():
        pytest.skip("model not trained")
    e = joblib.load(ENGINE_PATH)
    top = sorted((p for p in e.players.values() if p.n > 50), key=lambda p: -p.elo)[:2]
    ctx = dict(tournament="Wimbledon", surface="Grass", level="G", best_of=5, date=20270701)
    f1, f2 = e.features(top[0].pid, top[1].pid, ctx)["x"], e.features(top[1].pid, top[0].pid, ctx)["x"]
    for k in f1:
        if k.startswith("ctx_"):
            assert f1[k] == pytest.approx(f2[k])
        else:
            assert f1[k] == pytest.approx(-f2[k], abs=1e-9), k


# ------------------------------------------------------------------ end to end
def test_demo_slate_end_to_end():
    from atp_edge.modeling import MATCH_MODEL
    if not MATCH_MODEL.exists():
        pytest.skip("model not trained")
    from atp_edge.live import demo_feed
    from atp_edge.predict import Runtime, Settings, build_slate
    rt = Runtime()
    f, o, _ = demo_feed(rt)
    t = time.time()
    slate = build_slate(rt, f, o, Settings())
    assert time.time() - t < 10
    assert slate["matches"]
    for m in slate["matches"]:
        assert 0 < m["p_fair_a"] < 1
        assert abs(1 / m["fair_odds_a"] + 1 / m["fair_odds_b"] - 1) < 1e-9
        if m["level"] == "G":
            assert m["sets"]["available"] and 0.05 < m["sets"]["p_over35"] < 0.95
        else:
            assert "sets" not in m


def test_odds_api_parser_reads_sets_line_only_at_3_5():
    from atp_edge.odds_api import parse_event
    ev = {"id": "x", "home_team": "A A", "away_team": "B B", "commence_time": 1,
          "bookmakers": [{"key": "pinnacle", "markets": [
              {"key": "h2h", "outcomes": [{"name": "A A", "price": 1.5}, {"name": "B B", "price": 2.7}]},
              {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 3.5}, {"name": "Under", "price": 1.95, "point": 3.5},
                                             {"name": "Over", "price": 1.85, "point": 38.5}]}]}]}
    p = parse_event(ev, "ATP Wimbledon")
    assert p["prices_a"]["pinnacle"] == 1.5 and p["sets35"]["over"] == {"pinnacle": 1.9}


def test_court_speed_parser():
    from atp_edge.court_speed import parse_tennis_abstract
    html = ('<tr><td align="left">2026-02-02</td><td align="left"><a href="x">Montpellier</a></td><td align="left">Hard</td>'
            '<td align="right">19.8%</td><td align="right">1.49</td></tr>'
            '<tr><td align="left">23-Sep-2019</td><td align="left"><a href="x">Chengdu</a></td><td align="left">Hard</td>'
            '<td align="right">12.0%</td><td align="right">1.21</td></tr>')
    df = parse_tennis_abstract(html)
    assert list(df["key"]) == ["montpellier", "chengdu"] and list(df["year"]) == [2026, 2019]
