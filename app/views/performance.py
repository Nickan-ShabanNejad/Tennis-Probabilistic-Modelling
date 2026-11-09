import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.tracking import BETS_LOG  # noqa: E402
from common import TZ, kpi, setup  # noqa: E402

setup("Results & CLV")
st.markdown('<div class="hero"><h1>📈 Results & closing-line value</h1><div class="sub">Every bet the model signals is '
            'logged automatically before the match, with its Kelly stake, then settled from official results and '
            'compared with the closing price. The log is committed to GitHub, so it can\'t be edited afterwards.</div></div>',
            unsafe_allow_html=True)

bankroll = st.number_input("Bankroll for CA$ figures", min_value=0.0, value=float(st.session_state.get("bankroll", 1000.0)), step=100.0)

bets = pd.read_csv(BETS_LOG) if BETS_LOG.exists() else pd.DataFrame()
if not bets.empty and "strategy" in bets.columns:
    bets = bets[bets["strategy"].fillna("model") == "model"]
if bets.empty:
    st.info("No bets yet. The hourly job logs a bet whenever the model's price passes the EV and edge thresholds.")
    st.stop()

for c in ("result", "pl_units", "closing_price", "clv"):
    if c not in bets.columns:
        bets[c] = np.nan
bets["start"] = pd.to_datetime(bets["start_ts"], unit="s", utc=True).dt.tz_convert(TZ)
done = bets[bets["result"].isin(["won", "lost"])].sort_values("start_ts")
open_ = bets[~bets["result"].isin(["won", "lost", "void"])]
clv = pd.to_numeric(done["clv"], errors="coerce").dropna()

staked = float(done["stake"].sum())
pl = float(done["pl_units"].sum())
k = st.columns(4)
kpi(k[0], "Bets settled", f"{len(done):,}", f"{len(open_):,} open · {int((done['result'] == 'won').sum())} won")
kpi(k[1], "Profit", f"{pl:+.2%} of bankroll", f"CA${pl * bankroll:+,.2f}" if bankroll else "")
kpi(k[2], "ROI on stakes", f"{pl / staked:+.1%}" if staked else "—", f"{staked:.2%} of bankroll staked")
kpi(k[3], "Average CLV", f"{clv.mean():+.1%}" if len(clv) else "—",
    f"{(clv > 0).mean():.0%} beat the closing price" if len(clv) else "price taken vs closing price")

if len(clv) >= 2:
    se = clv.std(ddof=1) / np.sqrt(len(clv))
    z = clv.mean() / se if se > 0 else 0
    verdict = ("The bets are beating the closing price by more than two standard errors: evidence of a real edge."
               if z > 2 else "The bets are losing to the closing price by more than two standard errors: no edge so far."
               if z < -2 else f"Too early to tell ({len(clv)} bets, CLV {clv.mean():+.1%} ± {se:.1%}). A few hundred bets are needed.")
    st.caption("**Verdict:** " + verdict)
st.caption("CLV = price taken × closing no-vig probability − 1, using Pinnacle's last price before the start (or the "
           "books' consensus when Pinnacle has none). Consistently positive CLV is the earliest reliable sign of an edge; "
           "profit alone takes thousands of bets to separate skill from luck.")

if len(done):
    d = done.copy()
    d["cum_pl"] = d["pl_units"].cumsum() * 100
    d["cum_clv"] = pd.to_numeric(d["clv"], errors="coerce").expanding().mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["start"], y=d["cum_pl"], name="Cumulative profit (% bankroll)", mode="lines+markers",
                             line=dict(color="#3ddc84", width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=d["start"], y=d["cum_clv"], name="Running average CLV (%)", mode="lines",
                             line=dict(color="#3987e5", width=2, dash="dot")))
    fig.add_hline(y=0, line_color="#8b949e", line_dash="dot")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.15, x=0), hovermode="x unified",
                      yaxis=dict(gridcolor="rgba(139,148,158,.15)", ticksuffix="%"), xaxis=dict(gridcolor="rgba(139,148,158,.15)"))
    st.plotly_chart(fig, use_container_width=True)

    by = done.groupby("market").agg(bets=("bet_id", "size"), won=("result", lambda s: int((s == "won").sum())),
                                    staked=("stake", "sum"), profit=("pl_units", "sum"), clv=("clv", "mean"))
    by["roi"] = by["profit"] / by["staked"]
    st.dataframe(by.reset_index(), hide_index=True, use_container_width=True, column_config={
        "staked": st.column_config.NumberColumn("staked (bankroll)", format="percent"),
        "profit": st.column_config.NumberColumn("profit (bankroll)", format="percent"),
        "roi": st.column_config.NumberColumn(format="percent"), "clv": st.column_config.NumberColumn(format="percent")})

st.markdown("#### Bet log")
show = bets.sort_values("start_ts", ascending=False).copy()
show["start"] = show["start"].dt.strftime("%b %d %H:%M")
show["stake CA$"] = show["stake"] * bankroll
show["P/L CA$"] = pd.to_numeric(show["pl_units"], errors="coerce") * bankroll
cols = ["start", "tournament", "market", "selection", "price", "book", "probability", "ev", "stake", "stake CA$",
        "result", "P/L CA$", "closing_price", "clv"]
st.dataframe(show[[c for c in cols if c in show.columns]], hide_index=True, use_container_width=True, column_config={
    "probability": st.column_config.NumberColumn("model prob", format="percent"),
    "ev": st.column_config.NumberColumn(format="percent"), "stake": st.column_config.NumberColumn("stake (Kelly)", format="percent"),
    "stake CA$": st.column_config.NumberColumn(format="$%.2f"), "P/L CA$": st.column_config.NumberColumn(format="$%.2f"),
    "clv": st.column_config.NumberColumn(format="percent")})
st.download_button("Download bet log (CSV)", bets.drop(columns=["start"]).to_csv(index=False), "bets_log.csv", "text/csv")
