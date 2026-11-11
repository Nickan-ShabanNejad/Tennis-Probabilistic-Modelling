import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.tracking import BETS_LOG  # noqa: E402
from common import TZ, book_name, kpi, setup  # noqa: E402

setup("Results & CLV")
st.markdown('<div class="hero"><h1>📈 Results</h1><div class="sub">Is this actually working? Every bet the app signals is '
            'logged automatically before the match with its stake, settled from official results, and compared with the '
            'closing price. The log is committed to GitHub, so it can\'t be edited afterwards.</div></div>',
            unsafe_allow_html=True)

bankroll = st.number_input("Bankroll for CA$ figures", min_value=0.0, value=float(st.session_state.get("bankroll", 1000.0)), step=100.0)

bets = pd.read_csv(BETS_LOG) if BETS_LOG.exists() else pd.DataFrame()
RULES = {"blend": "Price first (current rule)", "model": "Model only (old rule)", "market": "Earliest version"}
if not bets.empty:
    if "strategy" not in bets.columns:
        bets["strategy"] = "model"
    bets["strategy"] = bets["strategy"].fillna("model")
    present = [r for r in RULES if (bets["strategy"] == r).any()]
    if len(present) > 1:
        pick = st.radio("Show bets flagged by", ["All rules"] + [RULES[r] for r in present], horizontal=True,
                        help="The betting rule changed in v2. Bets logged under the old rule are kept so the record stays honest; "
                             "judge the new rule only on its own bets.")
        if pick != "All rules":
            bets = bets[bets["strategy"].map(RULES) == pick]
if bets.empty:
    st.info("No bets yet. The live-prices job (every 10 minutes) logs a bet whenever a price passes the rule.")
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
kpi(k[3], "Beat the closing price by", f"{clv.mean():+.1%}" if len(clv) else "—",
    f"{(clv > 0).mean():.0%} of bets got a better price than the close" if len(clv) else "price taken vs closing price (CLV)")

if len(clv) >= 2:
    se = clv.std(ddof=1) / np.sqrt(len(clv))
    z = clv.mean() / se if se > 0 else 0
    verdict = ("The bets are beating the closing price by more than two standard errors: evidence of a real edge."
               if z > 2 else "The bets are losing to the closing price by more than two standard errors: no edge so far."
               if z < -2 else f"Too early to tell ({len(clv)} bets, CLV {clv.mean():+.1%} ± {se:.1%}). A few hundred bets are needed.")
    st.caption("**Verdict:** " + verdict)
st.caption("**How to read this.** Profit over a few dozen bets is mostly luck. The better early test is closing-line value "
           "(CLV): did you get a better price than the market's final price just before the match? CLV = price taken × closing "
           "fair probability − 1, using Pinnacle's last price before the start (or the books' consensus). Consistently positive "
           "CLV over a few hundred bets is the earliest reliable sign of a real edge.")

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

if not open_.empty:
    st.markdown("#### ⏳ Open bets: how the price has moved")
    o = open_.sort_values("start_ts").copy()
    o["start"] = o["start"].dt.strftime("%a %H:%M")
    for c in ("price_now", "book_now", "ev_now", "signal_now", "checked_at"):
        if c not in o.columns:
            o[c] = np.nan
    pn = pd.to_numeric(o["price_now"], errors="coerce")
    o["move"] = (pn / o["price"] - 1).where(pn.notna())
    o["status"] = [("✅ still a bet" if str(sig).lower() == "true" else "⚠️ no longer a bet") if pd.notna(pnow) else "–"
                   for sig, pnow in zip(o["signal_now"], pn)]
    o["stake CA$"] = o["stake"] * bankroll
    o["book"] = o["book"].map(book_name)
    o["book_now"] = o["book_now"].map(lambda b: book_name(b) if pd.notna(b) else None)
    st.dataframe(o[["start", "tournament", "selection", "price", "book", "price_now", "book_now", "move", "ev_now", "status", "stake", "stake CA$"]],
                 hide_index=True, use_container_width=True, column_config={
                     "price": st.column_config.NumberColumn("price when flagged", format="%.2f"),
                     "price_now": st.column_config.NumberColumn("best price now", format="%.2f"),
                     "book_now": st.column_config.TextColumn("book now"),
                     "move": st.column_config.NumberColumn("price move", format="percent"),
                     "ev_now": st.column_config.NumberColumn("EV now", format="percent"),
                     "stake": st.column_config.NumberColumn("stake (Kelly)", format="percent"),
                     "stake CA$": st.column_config.NumberColumn(format="$%.2f")})
    st.caption("Checked every 10 minutes until the match starts. Results and CLV always use the price when the bet "
               "was first flagged; if it has dropped, only bet it if 'still a bet' shows.")

st.markdown("#### Bet log")
show = bets.sort_values("start_ts", ascending=False).copy()
show["start"] = show["start"].dt.strftime("%b %d %H:%M")
show["stake CA$"] = show["stake"] * bankroll
show["book"] = show["book"].map(book_name)
show["rule"] = show["strategy"].map(RULES).fillna(show["strategy"])
show["P/L CA$"] = pd.to_numeric(show["pl_units"], errors="coerce") * bankroll
cols = ["start", "tournament", "market", "selection", "price", "book", "probability", "ev", "stake", "stake CA$",
        "result", "P/L CA$", "closing_price", "clv", "rule"]
st.dataframe(show[[c for c in cols if c in show.columns]], hide_index=True, use_container_width=True, column_config={
    "probability": st.column_config.NumberColumn("fair chance", format="percent"),
    "ev": st.column_config.NumberColumn(format="percent"), "stake": st.column_config.NumberColumn("stake (Kelly)", format="percent"),
    "stake CA$": st.column_config.NumberColumn(format="$%.2f"), "P/L CA$": st.column_config.NumberColumn(format="$%.2f"),
    "clv": st.column_config.NumberColumn(format="percent")})
st.download_button("Download bet log (CSV)", bets.drop(columns=["start"]).to_csv(index=False), "bets_log.csv", "text/csv")
