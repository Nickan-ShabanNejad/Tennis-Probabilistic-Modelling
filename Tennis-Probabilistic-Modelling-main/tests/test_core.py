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


# ------------------------------------------------------------------ v2: margin removal, market history, price-first rule
def test_no_vig_power_method_sums_to_one_and_leans_to_the_favourite():
    from atp_edge.betting import no_vig, no_vig_array, no_vig_proportional
    for a, b in ((1.9, 1.9), (1.25, 4.2), (1.05, 12.0), (2.1, 1.8), (1.3, 4.6)):
        assert no_vig(a, b) + no_vig(b, a) == pytest.approx(1.0, abs=1e-9)
    assert no_vig(1.9, 1.9) == pytest.approx(0.5)
    # the margin sits mostly on the outsider, so the favourite is rated higher than simple rescaling says
    assert no_vig(1.25, 4.2) > no_vig_proportional(1.25, 4.2)
    assert no_vig(None, 2.0) is None and no_vig(1.0, 2.0) is None
    arr = no_vig_array([1.5, float("nan"), 0.9], [2.5, 2.0, 2.0])
    assert arr[0] == pytest.approx(no_vig(1.5, 2.5)) and np.isnan(arr[1]) and np.isnan(arr[2])


def _toy_match(i, mkt=None, score="6-4 6-4", level="250", name="Brisbane"):
    r = dict(tourney_name=name, surface="Hard", tourney_level=level, tourney_date=20250101 + i,
             winner_id="a" if i % 2 else "b", loser_id="b" if i % 2 else "a", winner_name="A A", loser_name="B B",
             score=score, best_of=3, w_svpt=60, w_1stWon=30, w_2ndWon=12, l_svpt=62, l_1stWon=26, l_2ndWon=11)
    if mkt is not None:
        r["mkt_p_winner"] = mkt
    return r


def test_market_history_never_uses_the_price_of_the_match_being_predicted():
    """The closing line of a match may only influence predictions of LATER matches."""
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.engine import Engine
    e1, e2 = Engine(CourtSpeed(pd.DataFrame())), Engine(CourtSpeed(pd.DataFrame()))
    for i in range(5):
        e1.update(_toy_match(i, 0.6))
        e2.update(_toy_match(i, 0.6))
    ctx = dict(tournament="Brisbane", surface="Hard", level="250", best_of=3, date=20250110)
    before = e1.features("a", "b", ctx)["x"]
    assert before == e2.features("a", "b", ctx)["x"]
    assert before["mkt_offset_diff"] != 0
    # the next match is priced very differently in the two worlds: features made before it are identical...
    e1.update(_toy_match(5, 0.9))
    e2.update(_toy_match(5, 0.2))
    after1, after2 = e1.features("a", "b", ctx)["x"], e2.features("a", "b", ctx)["x"]
    # ...and only afterwards does the market's view show up.
    assert after1["mkt_offset_diff"] != after2["mkt_offset_diff"]
    assert {k: v for k, v in after1.items() if not k.startswith("mkt_")} == {k: v for k, v in after2.items() if not k.startswith("mkt_")}


def test_market_offset_fades_when_matches_have_no_price():
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.engine import Engine
    e = Engine(CourtSpeed(pd.DataFrame()))
    for i in range(4):
        e.update(_toy_match(i, 0.85))
    held = abs(e.players["a"].mkt_off)
    assert held > 0
    for i in range(4, 10):
        e.update(_toy_match(i, None, level="C", name="Some Challenger"))
    assert abs(e.players["a"].mkt_off) < held * 0.6


def test_serve_and_return_ratings_move_the_right_way():
    from atp_edge.court_speed import CourtSpeed
    from atp_edge.engine import Engine
    e = Engine(CourtSpeed(pd.DataFrame()))
    for i in range(8):   # "a" always wins 80% on serve, "b" 55%
        e.update(dict(tourney_name="Brisbane", surface="Hard", tourney_level="250", tourney_date=20250101 + i,
                      winner_id="a", loser_id="b", winner_name="A A", loser_name="B B", score="6-2 6-2", best_of=3,
                      w_svpt=50, w_1stWon=30, w_2ndWon=10, l_svpt=60, l_1stWon=22, l_2ndWon=11))
    a, b = e.players["a"], e.players["b"]
    assert a.srv[0] > 0 > b.srv[0] and a.rtn[0] > 0 > b.rtn[0]
    assert abs(a.srv[1]) > abs(a.srv[0])            # the fast rating reacts more
    assert a.games > 0 > b.games
    f = e.features("a", "b", dict(tournament="Brisbane", surface="Hard", level="250", best_of=3, date=20250201))["x"]
    assert f["srv_rating_diff"] > 0 and f["rtn_rating_diff"] > 0 and f["games_rating_diff"] > 0


def test_every_feature_belongs_to_a_family():
    from atp_edge.modeling import MATCH_MODEL
    if not MATCH_MODEL.exists():
        pytest.skip("model not trained")
    import joblib
    from atp_edge.predict import FAMILY_OF
    feats = [f for f in joblib.load(MATCH_MODEL).features if not f.startswith("ctx_")]
    assert not [f for f in feats if f not in FAMILY_OF]


def test_market_history_reads_the_live_price_log(tmp_path, monkeypatch):
    from atp_edge import market_history as mh
    log = tmp_path / "predictions_log.csv"
    ts = pd.Timestamp("2031-03-05 14:00").timestamp()
    pd.DataFrame([{"player_a": "Carlos Alcaraz", "player_b": "Jannik Sinner", "start_ts": ts, "p_market_a": 0.40,
                   "pinnacle_a": 2.40, "pinnacle_b": 1.60}]).to_csv(log, index=False)
    monkeypatch.setattr(mh, "PRED_LOG", log)
    monkeypatch.setattr(mh, "load_odds", lambda: pd.DataFrame())
    matches = pd.DataFrame([
        dict(tourney_name="Indian Wells", tourney_level="M", tourney_date=20310303, winner_name="Jannik Sinner", loser_name="Carlos Alcaraz"),
        dict(tourney_name="Indian Wells", tourney_level="M", tourney_date=20310303, winner_name="Taylor Fritz", loser_name="Ben Shelton"),
    ])
    out, report = mh.attach_market(matches)
    from atp_edge.betting import no_vig
    assert out["mkt_p_winner"].iloc[0] == pytest.approx(no_vig(1.60, 2.40))   # the winner was player_b in the log
    assert np.isnan(out["mkt_p_winner"].iloc[1])
    assert report["from_live_log"] == 1


def _priced_runtime():
    from atp_edge.modeling import MATCH_MODEL
    if not MATCH_MODEL.exists():
        pytest.skip("model not trained")
    from atp_edge.predict import Runtime
    rt = Runtime()
    top = sorted((p for p in rt.engine.players.values() if p.n > 50 and p.last_date and (rt.engine.last_date - p.last_date).days < 60),
                 key=lambda p: -p.elo)[:2]
    return rt, top[0].name, top[1].name


def test_price_first_rule_keeps_fair_price_close_to_the_sharp_market():
    from atp_edge.betting import no_vig
    from atp_edge.predict import Settings
    rt, a, b = _priced_runtime()
    prices_a = {"pinnacle": 1.80, "softbook": 1.84}
    prices_b = {"pinnacle": 2.10, "softbook": 2.02}
    m = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32", prices_a, prices_b, settings=Settings(strategy="blend"))
    market = no_vig(1.80, 2.10)
    assert m["fair_basis"] == "blend" and m["market_source"] == "Pinnacle"
    lo, hi = sorted((m["p_model_a"], market))
    assert lo - 1e-9 <= m["p_fair_a"] <= hi + 1e-9                       # between the model and the market...
    assert abs(m["p_fair_a"] - market) <= abs(m["p_model_a"] - market) * 0.5 + 1e-9   # ...and much nearer the market
    old = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32", prices_a, prices_b, settings=Settings(strategy="model"))
    assert old["fair_basis"] == "model" and old["p_fair_a"] == pytest.approx(old["p_model_a_injury_adj"])


def test_price_first_rule_flags_a_soft_price_and_gives_a_minimum_price():
    from atp_edge.predict import Settings
    rt, a, b = _priced_runtime()
    st = Settings(strategy="blend", min_ev=0.03, max_odds=8.0)
    base = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32", {"pinnacle": 1.90}, {"pinnacle": 1.95}, settings=st)
    fair_a = min(base["p_fair_a"], base["p_market_a"])
    generous = round((1 + 0.08) / fair_a, 2)                              # a book paying 8% over the fair price
    m = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32",
                   {"pinnacle": 1.90, "softbook": generous, "other": 1.88}, {"pinnacle": 1.95, "softbook": 1.80, "other": 1.93}, settings=st)
    rec = m["recommendation"]
    assert rec and rec["book"] == "softbook" and rec["ev"] > 0 and rec["ev_vs_sharp_price"] >= 0.03
    assert rec["min_price"] == pytest.approx(max(1.03 / rec["market_probability"], 1 / rec["probability"]))
    assert rec["min_price"] < rec["price"] and rec["next_best_price"] == pytest.approx(1.90)
    assert 0 < rec["stake"] <= st.kelly_cap
    # nothing to bet when every book is at or below the sharp price
    assert base["recommendation"] is None


def test_price_first_rule_needs_a_solid_reference_and_respects_max_odds():
    from atp_edge.predict import Settings
    rt, a, b = _priced_runtime()
    st = Settings(strategy="blend", min_ev=0.03, max_odds=8.0, min_books=3)
    # one lonely book and no Pinnacle: a probability is shown, but it is never a bet
    m = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32", {"softbook": 9.0}, {"softbook": 1.04}, settings=st)
    assert m["recommendation"] is None and m["market_reference_ok"] is False
    # a huge outsider price is skipped even when it looks like value
    m = rt.predict(a, b, "Paris Masters", time.time() + 3600, "R32",
                   {"pinnacle": 1.08, "softbook": 1.05}, {"pinnacle": 9.5, "softbook": 15.0}, settings=st)
    assert m["recommendation"] is None or m["recommendation"]["price"] <= 8.0


def test_backtest_reports_profit_and_drawdown():
    from atp_edge.betting import backtest
    p = np.array([0.6, 0.6, 0.6, 0.4])
    r = backtest(p, np.array([2.0, 2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0, 3.0]), min_ev=0.05)
    assert r["bets"] == 4 and r["max_drawdown_units"] >= 0
    assert r["profit_units"] == pytest.approx(r["kelly_roi"] * r["staked_units"])
    assert backtest(p, np.full(4, 1.5), np.full(4, 1.5), min_ev=0.05) == {"bets": 0}
    assert backtest(p, np.full(4, 2.0), np.array([2.0, 2.0, 2.0, 30.0]), min_ev=0.05, max_odds=8.0)["bets"] == 3
    # a second opinion that disagrees vetoes the bet
    assert backtest(p, np.full(4, 2.0), np.full(4, 2.0), min_ev=0.05, stake_p_winner=np.full(4, 0.5), min_stake_ev=0.01) == {"bets": 0}


def test_recent_seasons_weigh_more():
    from atp_edge.modeling import RECENCY_HALF_LIFE_YEARS, recency_weights
    w = recency_weights([2026, 2026 - RECENCY_HALF_LIFE_YEARS, 2000], 2026)
    assert w[0] == pytest.approx(1.0) and w[1] == pytest.approx(0.5) and w[2] < 0.05


def test_sets_market_follows_the_same_price_first_rule():
    from atp_edge.betting import no_vig
    from atp_edge.predict import Settings
    rt, a, b = _priced_runtime()
    st = Settings(strategy="blend", min_ev_sets=0.05)
    ml = ({"pinnacle": 1.50}, {"pinnacle": 2.70})
    # every book at or below Pinnacle's total-sets price: never a bet, whatever the sets model thinks
    flat = {"over": {"pinnacle": 1.95, "softbook": 1.90}, "under": {"pinnacle": 1.90, "softbook": 1.85}}
    m = rt.predict(a, b, "Australian Open", time.time() + 3600, "R32", *ml, sets35=flat, settings=st)
    s = m["sets"]
    assert s["available"] and s["recommendation"] is None
    market = no_vig(1.95, 1.90)
    lo, hi = sorted((s["p_over35"], market))
    assert s["p_market_over"] == pytest.approx(market) and lo - 1e-9 <= s["p_fair_over"] <= hi + 1e-9
    # a soft book paying well over the sharp price on the side the model also likes becomes a bet
    side = "over" if s["p_fair_over"] >= market else "under"
    p_side = market if side == "over" else 1 - market
    soft = {k: dict(v) for k, v in flat.items()}
    soft[side]["softbook"] = round(1.12 / min(p_side, s["p_fair_over"] if side == "over" else 1 - s["p_fair_over"]), 2)
    rec = rt.predict(a, b, "Australian Open", time.time() + 3600, "R32", *ml, sets35=soft, settings=st)["sets"]["recommendation"]
    assert rec and rec["book"] == "softbook" and rec["selection"].lower().startswith(side) and rec["min_price"] < rec["price"]
    assert 0 < rec["stake"] <= 0.015
