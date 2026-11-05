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
st.markdown('<div class="hero"><h1>🧠 Model & data</h1><div class="sub">Honest scoreboard: every number on this page is '
            'walk-forward. Each season is predicted only by a model trained on earlier seasons, then compared with '
            'closing odds.</div></div>', unsafe_allow_html=True)

b = m.get("market_benchmark", {})
if b:
    k = st.columns(4)
    kpi(k[0], "Model log loss", f"{b['model_log_loss']:.4f}", f"{b['matches']:,} matches with Pinnacle odds")
    kpi(k[1], "Pinnacle log loss", f"{b['pinnacle_log_loss']:.4f}", "closing price")
    kpi(k[2], "Accuracy", f"{b['model_accuracy']:.1%}", f"Pinnacle {b['pinnacle_accuracy']:.1%}")
    kpi(k[3], "Model weight on top of market", f"{m['market_blend']['model_coef']:.2f}", "0 = adds nothing measurable")
    py = pd.DataFrame(b.get("per_year", []))
    if not py.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=py["year"], y=py["model_log_loss"], name="Model", mode="lines+markers", line=dict(color="#3987e5", width=2), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=py["year"], y=py["market_log_loss"], name="Closing market", mode="lines+markers", line=dict(color="#d95926", width=2), marker=dict(size=8)))
        fig.update_layout(title="Log loss by season (lower is better)", height=320, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                          xaxis=dict(dtick=1, gridcolor="rgba(139,148,158,.15)"), yaxis=dict(tickformat=".3f", gridcolor="rgba(139,148,158,.15)"))
        st.plotly_chart(fig, use_container_width=True)

bt = m.get("backtests", {})
if bt:
    labels = {"model_only_ev2_edge2": "Model only · EV ≥ 2%, edge ≥ 2%", "model_only_ev5_edge5": "Model only · EV ≥ 5%, edge ≥ 5%",
              "market_anchored_ev2": "Market-anchored · EV ≥ 2% at Pinnacle"}
    rows = [{"Rule": labels.get(k, k), **v} for k, v in bt.items()]
    st.markdown("**Betting rules replayed at Pinnacle closing prices**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        c: st.column_config.NumberColumn(format="percent") for c in ("share_of_matches", "win_rate", "avg_claimed_ev", "flat_roi", "flat_roi_se", "kelly_roi")})
    st.caption("Closing prices are the hardest test. The realistic edge is a book whose price lags the fair price; the "
               "board shows the best price across all books The Odds API returns.")

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

st.markdown("### Features the model uses")
st.markdown(
    "- **Ratings:** dynamic-K Elo, overall and per surface, weighted by margin of victory and event level.\n"
    "- **Serve & return:** surface-specific serve and return points won, ace and double-fault rates.\n"
    "- **Form & fatigue:** last-10 results, Elo trend, matches, sets and minutes played in the last 7 to 14 days, rest days.\n"
    "- **Injuries:** retirements and walkovers (weighted toward recent ones), layoff length and comeback matches. Manual flags in "
    "`data/reference/injuries.csv` are applied on top.\n"
    "- **Event history:** results at this tournament compared with what the player's rating predicted.\n"
    "- **Event type:** the same comparison by level (Slam, Masters, 500, 250, Finals) and for best-of-five.\n"
    "- **Court speed:** the court speed index, each player's measured preference for faster or slower courts, and how "
    "serve, return and aces interact with speed.\n"
    "- **Head to head:** shrunk toward neutral for small samples, overall and per surface.")

status_path = GENERATED / "data_status.json"
with st.expander("Last data refresh"):
    st.json(json.loads(status_path.read_text()) if status_path.exists() else {"info": "no refresh yet"})
with st.expander("Manual injury flags (data/reference/injuries.csv)"):
    st.code((REFERENCE / "injuries.csv").read_text() if (REFERENCE / "injuries.csv").exists() else "", language="text")
with st.expander("Raw metrics"):
    st.json(m)
