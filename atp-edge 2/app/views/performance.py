import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.tracking import BETS_LOG, PRED_LOG, summary  # noqa: E402
from common import kpi, setup  # noqa: E402

setup("Results & CLV")
st.markdown('<div class="hero"><h1>📈 Results & closing-line value</h1><div class="sub">Every prediction and every '
            'recommended bet is logged automatically by GitHub Actions before the match starts, then settled from '
            'official results. The log is committed to the repository, so it can\'t be quietly edited afterwards.</div></div>',
            unsafe_allow_html=True)
s = summary()
k = st.columns(4)
kpi(k[0], "Predictions logged", f"{s.get('predictions', 0):,}", f"{s.get('settled', 0):,} settled")
kpi(k[1], "Bets settled", f"{s.get('settled_bets', 0):,}", f"{s.get('bets', 0):,} flagged in total")
kpi(k[2], "ROI", f"{s['roi']:+.1%}" if s.get("roi") is not None else "—", f"{s.get('units_pl', 0):+.2f} units of bankroll")
kpi(k[3], "Average CLV", f"{s['avg_clv']:+.1%}" if s.get("avg_clv") == s.get("avg_clv") and s.get("avg_clv") is not None else "—",
    "price taken vs Pinnacle close")
st.caption("CLV (closing-line value) is the best early test of skill. Results take thousands of bets to separate luck from "
           "edge; beating the closing price consistently shows up much sooner.")

st.markdown("#### Closing-line value")
c = st.columns(4)
kpi(c[0], "Bets that beat the close", f"{s['beat_close_rate']:.0%}" if s.get("beat_close_rate") is not None else "—",
    "share of settled bets with positive CLV")
kpi(c[1], "Model-lean CLV", f"{s['lean_clv']:+.2%}" if s.get("lean_clv") is not None else "—",
    f"± {s['lean_clv_se']:.2%} · {s['lean_clv_n']:,} matches" if s.get("lean_clv_se") else "every match, model's side at the opening price")
kpi(c[2], "Leans that beat the close", f"{s['lean_beat_close_rate']:.0%}" if s.get("lean_beat_close_rate") is not None else "—",
    "market moved toward the model")
verdict = "—"
if s.get("lean_clv") is not None and s.get("lean_clv_se"):
    z = s["lean_clv"] / s["lean_clv_se"]
    verdict = "edge" if z > 2 else ("no edge" if z < -2 else "too early")
kpi(c[3], "Verdict", verdict, "needs CLV > 2 standard errors above 0")
st.caption("Model-lean CLV takes the side the model preferred when a match first appeared, at Pinnacle's opening "
           "price, and asks whether the closing price moved that way. It's measured on every match, not just bets, "
           "so it's the fastest honest test of whether the model sees something the market doesn't. Prices are "
           "snapshotted every hour, and the last snapshot before the start is used as the close.")

if s.get("settled"):
    c = st.columns(3)
    c[0].metric("Model log loss (live)", f"{s['model_log_loss']:.3f}" if s.get("model_log_loss") else "—")
    c[1].metric("Fair-price log loss (live)", f"{s['fair_log_loss']:.3f}" if s.get("fair_log_loss") else "—")
    c[2].metric("Market log loss (live)", f"{s['market_log_loss']:.3f}" if s.get("market_log_loss") else "—")

bets = pd.read_csv(BETS_LOG) if BETS_LOG.exists() else pd.DataFrame()
if not bets.empty and "pl_units" in bets.columns:
    done = bets[bets["result"].isin(["won", "lost"])].copy()
    if len(done):
        done = done.sort_values("start_ts")
        done["cum"] = done["pl_units"].cumsum()
        fig = go.Figure(go.Scatter(x=pd.to_datetime(done["start_ts"], unit="s"), y=done["cum"] * 100, mode="lines",
                                   line=dict(color="#3ddc84", width=2), hovertemplate="%{x|%b %d}: %{y:+.2f}% of bankroll<extra></extra>"))
        fig.update_layout(title="Cumulative profit (% of bankroll)", height=300, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(gridcolor="rgba(139,148,158,.15)"), xaxis=dict(gridcolor="rgba(139,148,158,.15)"))
        st.plotly_chart(fig, use_container_width=True)
        by = done.groupby("market").agg(bets=("bet_id", "size"), staked=("stake", "sum"), pl=("pl_units", "sum"), clv=("clv", "mean"))
        by["roi"] = by["pl"] / by["staked"]
        st.dataframe(by.reset_index(), hide_index=True, use_container_width=True,
                     column_config={"roi": st.column_config.NumberColumn(format="percent"), "clv": st.column_config.NumberColumn(format="percent")})

st.markdown("#### Bet log")
if bets.empty:
    st.info("No bets yet. The log fills automatically once the slate workflow runs with your API keys.")
else:
    show = bets.copy()
    show["start"] = pd.to_datetime(show["start_ts"], unit="s", utc=True).dt.tz_convert("America/Toronto").dt.strftime("%b %d %H:%M")
    cols = [c for c in ["start", "tournament", "market", "selection", "price", "book", "ev", "stake", "result", "pl_units", "closing_price", "clv"] if c in show.columns]
    st.dataframe(show.sort_values("start_ts", ascending=False)[cols], hide_index=True, use_container_width=True,
                 column_config={"ev": st.column_config.NumberColumn(format="percent"), "stake": st.column_config.NumberColumn(format="percent"),
                                "pl_units": st.column_config.NumberColumn("P/L (bankroll)", format="percent"),
                                "clv": st.column_config.NumberColumn(format="percent")})
    st.download_button("Download bet log (CSV)", bets.to_csv(index=False), "bets_log.csv", "text/csv")

preds = pd.read_csv(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame()
if not preds.empty and "lean_clv" in preds.columns:
    lc = preds.dropna(subset=["lean_clv"]).sort_values("start_ts")
    lc = lc[lc["start_ts"] < pd.Timestamp.now(tz="UTC").timestamp()]
    if len(lc) >= 5:
        lc["running"] = lc["lean_clv"].expanding().mean() * 100
        fig = go.Figure(go.Scatter(x=pd.to_datetime(lc["start_ts"], unit="s"), y=lc["running"], mode="lines",
                                   line=dict(color="#3987e5", width=2), hovertemplate="%{x|%b %d}: %{y:+.2f}%<extra></extra>"))
        fig.add_hline(y=0, line_color="#8b949e", line_dash="dot")
        fig.update_layout(title="Running average model-lean CLV (%)", height=280, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(gridcolor="rgba(139,148,158,.15)"), xaxis=dict(gridcolor="rgba(139,148,158,.15)"))
        st.plotly_chart(fig, use_container_width=True)
if not preds.empty:
    with st.expander(f"Prediction log ({len(preds):,} matches)"):
        st.dataframe(preds.sort_values("start_ts", ascending=False), hide_index=True, use_container_width=True)
