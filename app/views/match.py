import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.betting import kelly_stake  # noqa: E402
from atp_edge.narrative import cached_preview, template_preview  # noqa: E402
from common import badge, current_slate, esc, level_name, local_time, settings_sidebar, setup  # noqa: E402

setup("Match detail")
settings = settings_sidebar()
slate, label, demo = current_slate(settings)
matches = slate["matches"]
if not matches:
    st.info("No upcoming matches. Go back to the board.")
    st.stop()

options = {f"{m['player_a']} vs {m['player_b']} · {m['event']} · {local_time(m.get('start_ts'))}": m for m in matches}
pre = st.session_state.get("match_detail")
default = 0
if pre:
    for i, m in enumerate(options.values()):
        if m["player_a"] == pre.get("player_a") and m["player_b"] == pre.get("player_b"):
            default = i
choice = st.selectbox("Match", list(options), index=default)
m = options[choice]
a, b = m["player_a"], m["player_b"]
la, lb = a.split()[-1], b.split()[-1]

st.markdown(f'<div class="hero"><h1>{esc(a)} vs {esc(b)}</h1><div class="sub">{esc(m["event"])} · {level_name(m["level"])} · '
            f'{esc(m["surface"])}{" indoor" if m.get("indoor") else ""} · best of {m["best_of"]} · {esc(local_time(m.get("start_ts")))}'
            f'</div></div>' + (badge("DEMO · synthetic prices", "bad") if demo else ""), unsafe_allow_html=True)

r = m.get("recommendation")
if r:
    st.success(f"**BET {r['selection']} @ {r['price']:.2f}** ({r.get('book')}) · model {r['probability']:.1%} · EV {r['ev']:+.1%} · "
               f"edge {r['edge_vs_market']:+.1%} · stake {r['stake']:.2%} of bankroll (¼-Kelly)")
elif m.get("best_a"):
    st.info(f"No bet: no price passes EV ≥ {settings.min_ev:.0%} and edge ≥ {settings.min_edge:.0%}.")
else:
    st.info("No bookmaker prices yet; the probabilities below are the model's.")

pm = m.get("p_market_a")
c = st.columns(5)
c[0].metric(f"Model: {la}", f"{m['p_model_a']:.1%}")
c[1].metric(f"Model: {lb}", f"{1 - m['p_model_a']:.1%}")
c[2].metric(f"Market: {la}", f"{pm:.1%}" if pm is not None else "—", help=f"No-vig probability from {m.get('market_source') or 'the books'}")
c[3].metric(f"Edge on {la}", f"{m['p_model_a'] - pm:+.1%}" if pm is not None else "—")
c[4].metric("Fair odds", f"{m['fair_odds_a']:.2f} / {m['fair_odds_b']:.2f}")

ai = cached_preview(m)
with st.container(border=True):
    st.markdown(("**🤖 AI preview**" if ai else "**📝 Preview**") + "<br>" + esc(ai or template_preview(m)), unsafe_allow_html=True)
    if not ai:
        st.caption("Add an ANTHROPIC_API_KEY secret in GitHub to get AI-written previews; this one is generated from the model's numbers.")

tab_why, tab_stats, tab_prices, tab_ctx = st.tabs(["Why the model leans", "Stats comparison", "Prices", "Context"])

with tab_why:
    f = m.get("factors") or {}
    if f:
        items = list(f.items())[::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                               marker_color=["#3987e5" if v >= 0 else "#d95926" for _, v in items],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", xaxis_title=f"← favours {lb}   ·   favours {la} →",
                          xaxis=dict(zeroline=True, zerolinecolor="#555", gridcolor="rgba(139,148,158,.15)"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each bar is that group's push on the log-odds. Serve covers service points won, aces and double faults; "
                   "Return covers return points won. Court-speed interactions are counted under Court speed only.")

with tab_stats:
    sa, sb = m.get("stats_a") or {}, m.get("stats_b") or {}
    lower_is_better = ("ATP rank", "Double faults", "Unforced errors")
    rows = []
    for key in sa:
        va, vb = sa.get(key), sb.get(key)
        is_rate = 0 < (va or 0) <= 1 or 0 < (vb or 0) <= 1
        fmt = (lambda v: "—" if v is None else f"{v:.1%}") if is_rate and "Elo" not in key and "rank" not in key and "Matches" not in key else (lambda v: "—" if v is None else f"{v:,.0f}")
        edge = ""
        if va is not None and vb is not None and fmt(va) != fmt(vb):
            better_a = (va < vb) if key.startswith(lower_is_better) else (va > vb)
            edge = f"◀ {la}" if better_a else f"{lb} ▶"
        rows.append({"Stat": key, a: fmt(va), "Better": edge, b: fmt(vb)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=min(40 + 35 * len(rows), 640))
        st.caption("Rates are averages over each player's recent matches with full statistics. Winners and unforced "
                   "errors are only published for some matches, so they can be missing.")

with tab_prices:
    pa_, pb_ = m.get("prices_a") or {}, m.get("prices_b") or {}
    books = sorted(set(pa_) | set(pb_))
    if books:
        p = m["p_fair_a"]
        tbl = pd.DataFrame([{"Book": k, a: pa_.get(k), f"EV {la}": (p * pa_[k] - 1) if pa_.get(k) else None,
                             b: pb_.get(k), f"EV {lb}": ((1 - p) * pb_[k] - 1) if pb_.get(k) else None,
                             "Margin": (1 / pa_[k] + 1 / pb_[k] - 1) if pa_.get(k) and pb_.get(k) else None} for k in books])
        st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
            f"EV {la}": st.column_config.NumberColumn(format="percent"), f"EV {lb}": st.column_config.NumberColumn(format="percent"),
            "Margin": st.column_config.NumberColumn("Bookmaker margin", format="percent")})
    else:
        st.info("No prices yet.")
    with st.container(border=True):
        st.markdown("**💱 Check your price**")
        x1, x2 = st.columns(2)
        side = x1.radio("Player", [a, b], key=f"ck_side_{choice}")
        p = m["p_fair_a"] if side == a else 1 - m["p_fair_a"]
        mk = None if pm is None else (pm if side == a else 1 - pm)
        dflt = (m.get("best_a") if side == a else m.get("best_b")) or round(1 / p, 2)
        price = x2.number_input("Odds", 1.01, 100.0, float(dflt), 0.01, key=f"ck_odds_{choice}_{side}")
        e = p * price - 1
        stake = kelly_stake(p, price, settings.kelly_fraction, settings.kelly_cap)
        edge_txt = f" · edge {p - mk:+.1%}" if mk is not None else ""
        if e >= settings.min_ev and (mk is None or p - mk >= settings.min_edge):
            st.success(f"Signal: EV {e:+.1%}{edge_txt}. Stake {stake:.2%} of bankroll.")
        else:
            st.warning(f"No signal: EV {e:+.1%}{edge_txt}. Need at least {(1 + settings.min_ev) / p:.2f} for EV ≥ {settings.min_ev:.0%}.")

with tab_ctx:
    sp = m["court_speed"]
    st.markdown(
        f"- **Court speed index:** {sp:.2f} ({'fast' if sp >= 1.15 else 'slow' if sp < 0.9 else 'medium'}; 1.00 = tour average) · "
        f"source: {esc(m['court_speed_source'])}" + (f", {m['court_speed_live_matches']} matches already played this week" if m.get("court_speed_live_matches") else "") + "\n"
        f"- **History at this event:** {esc(a)}: {esc(m['event_record_a'])} · {esc(b)}: {esc(m['event_record_b'])}\n"
        f"- **Head to head:** {m['h2h'][0]}–{m['h2h'][1]} overall, {m['h2h_surface'][0]}–{m['h2h_surface'][1]} on {esc(m['surface'])}\n"
        f"- **Ranking:** {int(m['rank_a'])} vs {int(m['rank_b'])}\n"
        f"- **Injury signals:** {esc(a)}: {esc(m['injury_signals_a'] or 'none')}{' · manual flag ' + str(m['injury_flag_a']) if m.get('injury_flag_a') else ''} · "
        f"{esc(b)}: {esc(m['injury_signals_b'] or 'none')}{' · manual flag ' + str(m['injury_flag_b']) if m.get('injury_flag_b') else ''}")
    sets = m.get("sets") or {}
    if sets.get("available"):
        st.markdown(f"**Total sets 3.5:** P(4+ sets) {sets['p_over35']:.0%} · fair over {sets['fair_over']:.2f} / under {sets['fair_under']:.2f}"
                    + (f" · signal {sets['recommendation']['selection']} @ {sets['recommendation']['price']:.2f}" if sets.get("recommendation") else ""))
