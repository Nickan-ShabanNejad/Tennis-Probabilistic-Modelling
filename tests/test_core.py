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
    ("BNP Paribas Nordic Open - Stockholm", "stockholm", "250"), ("BNP Paribas Open - Indian Wells", "indian wells", "M"),
    ("Rolex Paris Masters - Paris", "paris masters", "M"), ("ATP Finals", "atp finals", "F"), ("finals", "atp finals", "F"), ("Mubadala Citi DC Open - Washington", "washington", "500"),
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
    # ITF removed; Challenger kept for ratings (level "C", never bet on); qualifying removed
    assert len(out) == 2 and rep["lower_tier_removed"] == 1 and rep["qualifying_removed"] == 1
    assert set(out["tourney_level"]) == {"C", "250"}


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


def test_clv_tracking_keeps_opening_price_and_measures_close(tmp_path, monkeypatch):
    from atp_edge import tracking
    monkeypatch.setattr(tracking, "PRED_LOG", tmp_path / "p.csv")
    monkeypatch.setattr(tracking, "BETS_LOG", tmp_path / "b.csv")
    start = time.time() + 7200
    def slate(pa, pb):
        return {"matches": [{"player_a": "Carlos Alcaraz", "player_b": "Jannik Sinner", "start_ts": start, "event": "Paris Masters",
                             "level": "M", "surface": "Hard", "round": "F", "p_model_a": 0.60, "p_fair_a": 0.55, "p_market_a": 0.5,
                             "pinnacle_a": pa, "pinnacle_b": pb, "strategy": "market",
                             "recommendation": {"market": "Moneyline", "selection": "Carlos Alcaraz", "price": pa, "book": "pinnacle",
                                                "probability": 0.55, "ev": 0.05, "stake": 0.01}}]}
    tracking.record(slate(2.00, 1.90))   # open: Alcaraz 2.00
    tracking.record(slate(1.80, 2.10))   # later snapshot: market moved toward Alcaraz
    p = pd.read_csv(tmp_path / "p.csv")
    assert len(p) == 1 and p.loc[0, "open_pinnacle_a"] == 2.00 and p.loc[0, "pinnacle_a"] == 1.80
    assert p.loc[0, "model_lean"] == "Carlos Alcaraz" and p.loc[0, "lean_clv"] > 0.05
    b = pd.read_csv(tmp_path / "b.csv")
    assert len(b) == 1 and b.loc[0, "price"] == 2.00     # price taken is never overwritten
    m = pd.DataFrame([{"tourney_date": int(pd.Timestamp.now().strftime("%Y%m%d")), "winner_name": "Jannik Sinner",
                       "loser_name": "Carlos Alcaraz", "score": "7-6(5) 6-4"}])
    tracking.settle(m)
    b = pd.read_csv(tmp_path / "b.csv")
    assert b.loc[0, "result"] == "lost" and b.loc[0, "clv"] > 0.05 and b.loc[0, "pl_units"] == -0.01


def test_replay_handles_missing_best_of():
    """Matchstat rows have no best_of (NaN) - this crashed the first live retrain."""
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.data import clean
    from atp_edge.engine import Engine
    from atp_edge.history import replay
    df = pd.DataFrame([_row(tourney_name="Paris Masters", tourney_level="M", winner_id=f"w{i}", loser_id=f"l{i}",
                            winner_name=f"Ann W{i}", loser_name=f"Bob L{i}", best_of=np.nan, data_source="Matchstat")
                       for i in range(3)])
    rows = replay(df, Engine(CourtSpeed(pd.DataFrame())), progress=False)
    assert len(rows) == 3 and set(rows["_best_of"]) == {3}
    out, _ = clean(df.assign(tourney_name="Wimbledon", tourney_level="G"))
    assert set(out["best_of"]) == {5}


def test_matchstat_odds_parser_shapes():
    from atp_edge.matchstat_odds import parse_moneylines, parse_sets35
    nested = {"result": {"Pinnacle": {"Full Time Result": [{"od1": 1.9, "od2": 1.95, "sourceAddTime": 1},
                                                            {"od1": 1.8, "od2": 2.05, "sourceAddTime": 2}],
                                      "Total Sets 3.5": [{"od1": 1.7, "od2": 2.1, "sourceAddTime": 2}]},
                         "William Hill": {"Match Winner": {"start": {"od1": 1.75, "od2": 2.0}}}}}
    ml = parse_moneylines(nested)
    assert ml == {"pinnacle": (1.8, 2.05), "williamhill": (1.75, 2.0)}
    assert parse_sets35(nested)["over"] == {"pinnacle": 1.7}
    flat = [{"bookmaker": "Bet365", "market": "Match Winner", "od1": 1.5, "od2": 2.6},
            {"bookmaker": "Bet365", "market": "Set Handicap", "od1": 1.9, "od2": 1.9}]
    assert parse_moneylines(flat) == {"bet365": (1.5, 2.6)}


def test_matchstat_odds_real_payload():
    """Response captured from the live Matchstat pre-match odds endpoint (Stockholm, Nov 2026)."""
    import json
    from atp_edge.matchstat_odds import parse_moneylines
    payload = json.loads((Path(__file__).parent / "matchstat_odds_sample.json").read_text())["payload"]
    ml = parse_moneylines(payload)
    assert ml["pinnacle"] == (1.323, 3.66) and ml["bet365"] == (1.3, 3.5) and len(ml) == 4


# ------------------------------------------------------------------ live prices
def test_dedupe_fills_gaps_from_duplicate():
    from atp_edge.data import KEEP_COLS, dedupe
    base = {c: np.nan for c in KEEP_COLS}
    a = {**base, "tourney_name": "Paris Masters", "tourney_date": 20261102, "winner_name": "Jannik Sinner",
         "loser_name": "Carlos Alcaraz", "score": "6-4 6-4", "data_source": "TennisMyLife", "w_ace": 9.0, "w_df": 1.0, "minutes": 90.0}
    b = {**base, "tourney_name": "Rolex Paris Masters", "tourney_date": 20261108, "winner_name": "J. Sinner",
         "loser_name": "C. Alcaraz", "score": "6-4 6-4", "data_source": "Matchstat", "w_ace": 9.0, "winner_rank": 1.0}
    out, n = dedupe(pd.DataFrame([a, b]))
    assert n == 1 and len(out) == 1
    assert out.loc[0, "w_df"] == 1.0 and out.loc[0, "winner_rank"] == 1.0   # gaps filled from the duplicate


def test_price_refresh_gets_faster_near_the_start(monkeypatch):
    from atp_edge.live import is_due, refresh_minutes
    monkeypatch.delenv("PRICE_INTERVAL_MINUTES", raising=False)
    assert refresh_minutes(1) == 10 and refresh_minutes(6) == 30 and refresh_minutes(30) == 120
    now = time.time()
    assert is_due(None, now + 3600, now)
    assert is_due(now - 11 * 60, now + 3600, now)          # 1 h out, 11 min old -> refresh
    assert not is_due(now - 11 * 60, now + 20 * 3600, now)  # 20 h out -> wait for 2 h


def test_matchstat_budget_spreads_quota_and_keeps_reserve():
    from atp_edge.live import matchstat_budget
    now = time.time()
    assert matchstat_budget({}, now) == 40
    assert matchstat_budget({"limit": 10000, "remaining": 200, "reset_s": 86400, "at": now}, now) == 0
    # 2,000 spare calls over 10 days of 10-minute runs -> about 1 call per run
    assert matchstat_budget({"limit": 2000, "remaining": 2300, "reset_s": 10 * 86400, "at": now}, now) == 1
    assert matchstat_budget({"limit": 100000, "remaining": 90000, "reset_s": 86400, "at": now}, now) == 40


def test_collect_only_pulls_prices_that_are_due(tmp_path, monkeypatch):
    from atp_edge import live, odds_api
    from atp_edge.tournaments import resolve
    monkeypatch.setattr(live, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setenv("ODDS_API_KEY", "test")
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    now = time.time()
    calls = []

    class FakeAPI:
        remaining, used = "15000", "5000"
        def __init__(self, *a, **k): pass
        def atp_sports(self):
            return [{"key": "tennis_atp_finals", "title": "ATP Finals", "event": resolve("ATP Finals")}]
        def odds(self, key, slam=False):
            calls.append(key)
            return [{"id": "e1", "home_team": "Jannik Sinner", "away_team": "Carlos Alcaraz", "commence_time": now + 3600,
                     "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                         {"name": "Jannik Sinner", "price": 1.8}, {"name": "Carlos Alcaraz", "price": 2.1}]}]}]}]
    monkeypatch.setattr(odds_api, "OddsAPI", FakeAPI)
    f, o, s = live.collect(None)
    assert calls == ["tennis_atp_finals"] and o[0]["prices_a"]["pinnacle"] == 1.8 and o[0]["prices_ts"]
    live.collect(None)                                   # 0 s later: cached, no credit spent
    assert len(calls) == 1
    monkeypatch.setattr(live.time, "time", lambda: now + 11 * 60)
    live.collect(None)                                   # match 1 h away, prices 11 min old: refresh
    assert len(calls) == 2


def test_logged_bet_tracks_line_moves(tmp_path, monkeypatch):
    from atp_edge import tracking
    monkeypatch.setattr(tracking, "PRED_LOG", tmp_path / "p.csv")
    monkeypatch.setattr(tracking, "BETS_LOG", tmp_path / "b.csv")
    start = time.time() + 7200
    def slate(price, rec):
        return {"matches": [{"player_a": "Carlos Alcaraz", "player_b": "Jannik Sinner", "start_ts": start, "event": "ATP Finals",
                             "level": "F", "surface": "Hard", "p_model_a": 0.55, "p_fair_a": 0.55, "p_market_a": 0.45,
                             "best_a": price, "best_book_a": "bet365", "best_b": 1.7, "best_book_b": "bet365", "strategy": "model",
                             "recommendation": {"market": "Moneyline", "selection": "Carlos Alcaraz", "price": price, "book": "bet365",
                                                "probability": 0.55, "ev": 0.55 * price - 1, "stake": 0.01} if rec else None}]}
    r = tracking.record(slate(2.20, True))
    assert r["new_bets"] == 1 and r["new_bet_rows"][0]["price"] == 2.20
    r = tracking.record(slate(1.85, False))              # price shortened, no longer a bet
    assert r["new_bets"] == 0
    b = pd.read_csv(tmp_path / "b.csv")
    assert b.loc[0, "price"] == 2.20 and b.loc[0, "price_now"] == 1.85
    assert not bool(b.loc[0, "signal_now"]) and b.loc[0, "ev_now"] == pytest.approx(0.55 * 1.85 - 1)


def test_slate_fingerprint_ignores_timestamps():
    sys.path.insert(0, str(ROOT / "scripts"))
    from live_loop import slate_fingerprint
    m = {"player_a": "A", "player_b": "B", "start_ts": 1, "p_model_a": 0.6, "prices_a": {"x": 1.5}, "prices_b": {"x": 2.6}}
    a = slate_fingerprint({"generated_at": "t1", "matches": [m]})
    assert a == slate_fingerprint({"generated_at": "t2", "matches": [dict(m)]})
    assert a != slate_fingerprint({"matches": [{**m, "prices_a": {"x": 1.45}}]})
