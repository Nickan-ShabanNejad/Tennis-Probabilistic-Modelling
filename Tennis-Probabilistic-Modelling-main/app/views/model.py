import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.config import GENERATED, REFERENCE  # noqa: E402
from atp_edge.court_speed import TA_PATH  # noqa: E402
from atp_edge.tournaments import EVENTS, level_label  # noqa: E402
from common import kpi, runtime, setup  # noqa: E402

setup("Model & data")
m = runtime().metrics
st.markdown('<div class="hero"><h1>🧠 Model & test results</h1><div class="sub">Every number on this page is walk-forward: each '
            'season is predicted only from earlier seasons, then compared with the bookmakers\' closing prices. Nothing here '
            'was tuned on the matches it is scored on.</div></div>', unsafe_allow_html=True)

GRID = "rgba(139,148,158,.15)"
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
              legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))

# ------------------------------------------------------------------ accuracy
b = m.get("market_benchmark", {})
bl = m.get("blend_benchmark", {})
if b:
    st.markdown("### How accurate is it?")
    k = st.columns(4)
    kpi(k[0], "Model picks the winner", f"{b['model_accuracy']:.1%}", f"{b['matches']:,} matches, {b['years'][0]}–{b['years'][-1]}")
    kpi(k[1], "Pinnacle's favourite wins", f"{b['pinnacle_accuracy']:.1%}", "the closing price, the hardest benchmark")
    kpi(k[2], "Model log loss", f"{b['model_log_loss']:.4f}", f"market {b['pinnacle_log_loss']:.4f} · lower is better")
    if bl.get("fair_log_loss"):
        kpi(k[3], "Fair price (market + model)", f"{bl['fair_log_loss']:.4f}",
            f"market alone {bl['market_log_loss']:.4f} on the same {bl['matches']:,} matches")
    st.markdown(
        '<div class="note"><b>Plain reading.</b> On its own the model is good but clearly less accurate than the closing price of a '
        'sharp bookmaker, as almost every public model is. Blending a little of the model into the market price is very slightly '
        'better than the market alone, a gap too small to be sure of. So the app does not bet because “the model disagrees with '
        'the market”; it bets when a bookmaker\'s price is out of line with the sharp price.</div>', unsafe_allow_html=True)
    py = pd.DataFrame(b.get("per_year", []))
    if not py.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=py["year"], y=py["model_log_loss"], name="Model", mode="lines+markers", line=dict(color="#3987e5", width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=py["year"], y=py["market_log_loss"], name="Closing market", mode="lines+markers", line=dict(color="#d95926", width=2), marker=dict(size=8)))
        fig.update_layout(title="Prediction error by season (log loss, lower is better)", height=320, **LAYOUT,
                          xaxis=dict(dtick=1, gridcolor=GRID), yaxis=dict(tickformat=".3f", gridcolor=GRID))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ betting rules
bt = m.get("backtests", {})
rules = bt.get("rules", {})
if rules:
    st.markdown("### Would the betting rule have made money?")
    st.caption(f"{bt['matches']:,} matches, {bt['years'][0]}–{bt['years'][1]}, each rule replayed on exactly the same matches at three "
               "prices. ROI = profit per dollar staked with equal stakes. ± is one standard error: a result within about two of "
               "them of zero could be luck. Claimed value and worst losing run are for the best-price column.")
    names = {"old_model_only": "Old rule: model only (EV ≥ 5%, edge ≥ 5%)", "new_ev3": "Current rule: price first, 3% (default)",
             "new_ev2": "Current rule at 2%", "new_ev5": "Current rule at 5%", "price_only_ev3": "Price shopping alone, no model (3%)"}
    def cell(r: dict) -> str:
        if not r.get("bets") or r["bets"] < 30:
            return "no bets"
        return f"{r['flat_roi']:+.1%} ± {r['flat_roi_se']:.1%}  ({r['bets']:,} bets)"

    rows = []
    for rule in ("old_model_only", "new_ev3", "new_ev2", "new_ev5", "price_only_ev3"):
        rr = rules.get(rule) or {}
        best_ = rr.get("best") or {}
        rows.append({"Rule": names[rule],
                     "ROI at Pinnacle's closing price": cell(rr.get("pinnacle") or {}),
                     "ROI halfway between average and best": cell(rr.get("halfway") or {}),
                     "ROI at the best price of any book": cell(best_),
                     "Value the rule claimed": f"{best_['avg_claimed_ev']:+.1%}" if best_.get("bets") else "—",
                     "Worst losing run": f"{best_['max_drawdown_units']:.2f} bankrolls" if best_.get("bets") else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    old, new = (rules.get("old_model_only") or {}), (rules.get("new_ev3") or {})
    if old.get("best", {}).get("bets") and new.get("best", {}).get("bets"):
        ob, nb, op = old["best"], new["best"], old.get("pinnacle", {})
        st.markdown(
            f'<div class="note ok"><b>What changed and why.</b> The old rule treated the model\'s number as the truth. It claimed '
            f'{ob["avg_claimed_ev"]:+.0%} value per bet and delivered {ob["flat_roi"]:+.1%} at the best price'
            + (f' ({op["flat_roi"]:+.1%} at Pinnacle)' if op.get("bets") else "")
            + f', with a worst losing run of {ob["max_drawdown_units"]:.1f} bankrolls because it staked the maximum on almost every '
            f'bet. The current rule claimed {nb["avg_claimed_ev"]:+.1%} and delivered {nb["flat_roi"]:+.1%} ± {nb["flat_roi_se"]:.1%}, '
            f'with a worst run of {nb["max_drawdown_units"]:.2f} bankrolls.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note"><b>Read this before trusting the best-price column.</b> “Best closing price of any book” is the '
        'highest price among dozens of bookmakers. You will not have accounts at all of them, the best price is often gone within '
        'minutes, and books limit customers who only take their top prices. Expect real results between the “halfway” and “best” '
        'columns, and fewer bets. At Pinnacle\'s own price the current rule finds nothing to bet: the edge is in price differences '
        'between books, not in out-predicting the market.</div>', unsafe_allow_html=True)
    per = pd.DataFrame([p for p in bt.get("by_period", []) if p.get("bets", 0) >= 30])
    if not per.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=per["year"], y=per["flat_roi"], name="ROI", marker_color=["#3ddc84" if v >= 0 else "#e66767" for v in per["flat_roi"]],
                             error_y=dict(type="data", array=per["flat_roi_se"], color="#8b949e", thickness=1.2),
                             customdata=per[["bets"]], hovertemplate="%{x}: %{y:+.1%} on %{customdata[0]} bets<extra></extra>"))
        fig.update_layout(title="Current rule by season, at the best price (bars: ROI, whiskers: ± one standard error)", height=320,
                          **{**LAYOUT, "hovermode": "closest"}, showlegend=False,
                          xaxis=dict(dtick=1, gridcolor=GRID), yaxis=dict(tickformat="+.0%", gridcolor=GRID, zerolinecolor="#555"))
        st.plotly_chart(fig, use_container_width=True)
        first, last = per.iloc[0], per.iloc[-1]
        st.caption(f"Qualifying bets fell from {int(per['bets'].max())} in the busiest season to {int(last['bets'])} in {int(last['year'])}: "
                   "bookmakers' prices have moved closer together, so opportunities are getting rarer. Not every season is positive.")

blend = m.get("market_blend", {})
if blend.get("available"):
    with st.expander("How the fair price is built (technical)"):
        st.markdown(
            f"`fair log-odds = {blend['model_coef']:.2f} × model log-odds + {blend['market_coef']:.2f} × market log-odds`, fitted on "
            f"{blend['fitted_matches']:,} matches with a Pinnacle closing price"
            + (f"; when Pinnacle is missing and the books' consensus is used instead, the weights are "
               f"{blend['consensus_model_coef']:.2f} and {blend['consensus_market_coef']:.2f}." if "consensus_model_coef" in blend else ".")
            + "\n\nThe market price has the bookmaker's margin removed with the *power* method, which puts more of the margin on "
              "outsiders (as bookmakers do) instead of spreading it evenly."
            + (f" On Pinnacle closing prices that scores {b['pinnacle_log_loss']:.4f} against {b['pinnacle_log_loss_simple_devig']:.4f} "
               "for the simple method." if b.get("pinnacle_log_loss_simple_devig") else ""))

st.markdown("### What the model looks at")
st.markdown(
    "- **Ratings:** dynamic-K Elo, overall, per surface and indoor/outdoor, weighted by margin of victory and event level; plus a "
    "rating built on share of games won.\n"
    "- **Serve & return:** opponent-adjusted serve and return ratings updated after every match (slow, fast and per surface), "
    "recent serve/return points won, ace and double-fault rates.\n"
    "- **Past betting lines:** how the closing line rated each player in his recent matches compared with his Elo. Only finished "
    "matches are used; today's price never feeds today's prediction.\n"
    "- **Form & fatigue:** last-10 results, Elo trend, matches, sets and minutes played in the last 7 to 14 days, rest days.\n"
    "- **Injuries:** retirements and walkovers (weighted toward recent ones), layoff length and comeback matches. Manual flags in "
    "`data/reference/injuries.csv` are applied on top.\n"
    "- **Event history and type:** results at this tournament, at this level and in best-of-five, against what the rating predicted.\n"
    "- **Court speed:** the court speed index, each player's measured preference for faster or slower courts, and how "
    "serve, return and aces interact with speed.\n"
    "- **Head to head:** shrunk toward neutral for small samples, overall and per surface.\n"
    f"- **Recent seasons count more:** a match from {m.get('recency_half_life_years', 5):.0f} seasons ago has half the weight of one from this season.")

st.markdown("### Court speed index")
if TA_PATH.exists():
    ta = pd.read_csv(TA_PATH)
    latest = ta.sort_values("year").groupby("key").tail(1)
    latest["event"] = latest["key"].map(lambda k: EVENTS[k].name if k in EVENTS else None)
    latest = latest.dropna(subset=["event"])
    latest["level"] = latest["key"].map(lambda k: level_label(EVENTS[k].level))
    st.caption("Latest Tennis Abstract surface-speed rating per event (1.00 = tour average). During an event the model "
               "blends this with the ace rates of matches already played that week.")
    st.dataframe(latest.sort_values("speed", ascending=False)[["event", "level", "surface", "year", "speed", "ace_pct"]],
                 hide_index=True, use_container_width=True, height=360,
                 column_config={"ace_pct": st.column_config.NumberColumn("ace %", format="percent"), "speed": st.column_config.NumberColumn(format="%.2f")})

status_path = GENERATED / "data_status.json"
with st.expander("Last data refresh"):
    mh = (m.get("data") or {}).get("market_history") or {}
    if mh:
        st.caption(f"Past betting lines: {mh.get('priced_matches', 0):,} matches priced, latest {mh.get('latest_priced_match')} "
                   f"({mh.get('from_live_log', 0)} from this app's own price log).")
    st.json(json.loads(status_path.read_text()) if status_path.exists() else {"info": "no refresh yet"})
with st.expander("Manual injury flags (data/reference/injuries.csv)"):
    st.code((REFERENCE / "injuries.csv").read_text() if (REFERENCE / "injuries.csv").exists() else "", language="text")
with st.expander("Raw metrics"):
    st.json(m)
