import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.betting import kelly_stake  # noqa: E402
from atp_edge.predict import judge_price  # noqa: E402
from atp_edge.narrative import cached_preview, template_preview  # noqa: E402
from common import age_text, badge, book_name, current_slate, esc, fmt_odds, level_name, local_time, money, settings_sidebar, setup  # noqa: E402

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

price_first = m.get("fair_basis") == "blend"
r = m.get("recommendation")
if r:
    floor = f" Don't take less than **{fmt_odds(r['min_price'])}**." if r.get("min_price") else ""
    st.success(f"**Bet {money(r['stake'])} on {r['selection']} at {fmt_odds(r['price'])}** ({book_name(r.get('book'))}). "
               f"Fair price {fmt_odds(r['fair_odds'])}, so this pays about {r['ev']:+.1%} per dollar staked on average.{floor}")
elif m.get("best_a"):
    va, vb = m.get("verdict_a") or {}, m.get("verdict_b") or {}
    need = (f" {la} would need **{fmt_odds(va.get('min_price'))}** (best now {fmt_odds(m.get('best_a'))}); "
            f"{lb} would need **{fmt_odds(vb.get('min_price'))}** (best now {fmt_odds(m.get('best_b'))})."
            if va.get("min_price") and vb.get("min_price") else "")
    st.info("**No bet.** No book's price is far enough above the fair price." + need)
else:
    st.info("No bookmaker prices yet; the probabilities below are the model's.")

pm = m.get("p_market_a")
pf = m["p_fair_a"]
c = st.columns(4)
c[0].metric(f"Model says {la}", f"{m['p_model_a']:.1%}", help="The rating model on its own: results, serve and return, form, fatigue, surface, court speed.")
c[1].metric(f"Market says {la}", f"{pm:.1%}" if pm is not None else "—",
            help=f"{m.get('market_source') or 'The books'} price with the bookmaker's margin removed.")
c[2].metric(f"Fair chance used · {la}", f"{pf:.1%}", delta=(f"{pf - pm:+.1%} vs market" if pm is not None and price_first else None), delta_color="off",
            help="Price first: the market's number, moved a little toward the model. This is what bets and stakes are based on."
            if price_first else "Model-only rule: the model's number is used as it is.")
c[3].metric("Fair prices", f"{fmt_odds(m['fair_odds_a'])} / {fmt_odds(m['fair_odds_b'])}", help=f"{la} / {lb}. A bet needs a book to pay clearly more than this.")
if pm is not None and abs(m["p_model_a"] - pm) >= 0.08:
    st.markdown(f'<div class="note">The model and the market disagree by {abs(m["p_model_a"] - pm):.0%} here. Over 18,000+ past matches the '
                f'market was right more often when that happened, usually because it knows something the model can\'t see '
                f'(fitness, motivation, late news). That is why the fair chance stays close to the market.</div>', unsafe_allow_html=True)

ai = cached_preview(m)
with st.container(border=True):
    st.markdown(("**🤖 AI preview**" if ai else "**📝 Preview**") + "<br>" + esc(ai or template_preview(m)), unsafe_allow_html=True)
    if not ai:
        st.caption("Add an ANTHROPIC_API_KEY secret in GitHub to get AI-written previews; this one is generated from the model's numbers.")

tab_prices, tab_why, tab_stats, tab_ctx = st.tabs(["Prices at every book", "Why the model leans", "Stats comparison", "Context"])

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
        st.caption("Longer bar = bigger push toward that player in the model's own estimate. “Past betting lines” is how the "
                   "market rated each player in his recent matches compared with his results-based rating. "
                   "Court-speed interactions are counted under Court speed only.")

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
        tbl = pd.DataFrame([{"Book": book_name(k), a: pa_.get(k), f"EV {la}": (p * pa_[k] - 1) if pa_.get(k) else None,
                             b: pb_.get(k), f"EV {lb}": ((1 - p) * pb_[k] - 1) if pb_.get(k) else None,
                             "Margin": (1 / pa_[k] + 1 / pb_[k] - 1) if pa_.get(k) and pb_.get(k) else None} for k in books])
        tbl = tbl.sort_values(a, ascending=False)
        st.caption("EV = expected profit per dollar at that book's price, using the fair chance above. Positive is good; "
                   f"a bet needs the price to beat the sharp market by {settings.min_ev:.0%}.")
        st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
            f"EV {la}": st.column_config.NumberColumn(format="percent"), f"EV {lb}": st.column_config.NumberColumn(format="percent"),
            "Margin": st.column_config.NumberColumn("Bookmaker margin", format="percent")})
    else:
        st.info("No prices yet.")
    if m.get("prices_ts"):
        st.caption(f"Prices from {m.get('odds_source') or 'the books'}, updated {age_text(m['prices_ts'])}. "
                   "They refresh automatically: every 10 minutes in the last 3 hours before the start, "
                   "every 30 minutes before that, every 2 hours a day or more out.")
    with st.container(border=True):
        st.markdown("**💱 Check your price**")
        x1, x2 = st.columns(2)
        side = x1.radio("Player", [a, b], key=f"ck_side_{choice}")
        p = m["p_fair_a"] if side == a else 1 - m["p_fair_a"]
        mk = None if pm is None else (pm if side == a else 1 - pm)
        dflt = (m.get("best_a") if side == a else m.get("best_b")) or round(1 / p, 2)
        price = x2.number_input("Odds", 1.01, 100.0, float(dflt), 0.01, key=f"ck_odds_{choice}_{side}")
        v = judge_price(p, mk, price, settings, m.get("fair_basis", "model"), bool(m.get("market_reference_ok", mk is not None)))
        stake = kelly_stake(p, price, settings.kelly_fraction, settings.kelly_cap)
        if v["ok"]:
            st.success(f"**Bet.** {v['reason']} Stake {money(stake)}.")
        else:
            st.warning("**Not a bet.** " + v["reason"] + (f" You'd need {fmt_odds(v['min_price'])} or better." if v.get("min_price") else ""))

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
