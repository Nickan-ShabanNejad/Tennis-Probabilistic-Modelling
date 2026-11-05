import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from atp_edge.betting import kelly_stake  # noqa: E402
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

st.markdown(f'<div class="hero"><h1>{esc(a)} vs {esc(b)}</h1><div class="sub">{esc(m["event"])} · {level_name(m["level"])} · '
            f'{esc(m["surface"])}{" indoor" if m.get("indoor") else ""} · best of {m["best_of"]} · {esc(local_time(m.get("start_ts")))}'
            f'</div></div>' + (badge("DEMO · synthetic prices", "bad") if demo else ""), unsafe_allow_html=True)

r = m.get("recommendation")
if r:
    st.success(f"**BET {r['selection']} @ {r['price']:.2f}** ({r.get('book')}) · EV {r['ev']:+.1%} · fair odds {r['fair_odds']:.2f} · "
               f"stake {r['stake']:.2%} of bankroll")
elif m.get("best_a"):
    st.info("No moneyline value at the best available prices.")
else:
    st.info("No bookmaker prices yet; the probabilities below are the model's.")

c = st.columns(5)
c[0].metric(f"Fair P({a.split()[-1]})", f"{m['p_fair_a']:.1%}")
c[1].metric(f"Fair P({b.split()[-1]})", f"{1 - m['p_fair_a']:.1%}")
c[2].metric("Model only", f"{m['p_model_a']:.1%}", help="Raw model probability for the first player, before combining with the market")
c[3].metric(f"Market ({m.get('market_source') or '—'})", f"{m['p_market_a']:.1%}" if m.get("p_market_a") is not None else "—")
c[4].metric("Fair odds", f"{m['fair_odds_a']:.2f} / {m['fair_odds_b']:.2f}")

left, right = st.columns([1.15, 1])
with left:
    st.markdown("#### Why the model leans this way")
    f = m.get("factors") or {}
    if f:
        items = list(f.items())[::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                               marker_color=["#3987e5" if v >= 0 else "#d95926" for _, v in items],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", xaxis_title=f"← favours {b.split()[-1]}   ·   favours {a.split()[-1]} →",
                          xaxis=dict(zeroline=True, zerolinecolor="#555", gridcolor="rgba(139,148,158,.15)"))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Context")
    sp = m["court_speed"]
    st.markdown(
        f"- **Court speed index:** {sp:.2f} ({'fast' if sp >= 1.15 else 'slow' if sp < 0.9 else 'medium'}; 1.00 = tour average) · "
        f"source: {esc(m['court_speed_source'])}" + (f", {m['court_speed_live_matches']} matches already played this week" if m.get("court_speed_live_matches") else "") + "\n"
        f"- **History at this event:** {esc(a)}: {esc(m['event_record_a'])} · {esc(b)}: {esc(m['event_record_b'])}\n"
        f"- **Head to head:** {m['h2h'][0]}–{m['h2h'][1]} overall, {m['h2h_surface'][0]}–{m['h2h_surface'][1]} on {esc(m['surface'])}\n"
        f"- **Ranking:** {int(m['rank_a'])} vs {int(m['rank_b'])}\n"
        f"- **Injury signals:** {esc(a)}: {esc(m['injury_signals_a'] or 'none')}{' · manual flag ' + str(m['injury_flag_a']) if m.get('injury_flag_a') else ''} · "
        f"{esc(b)}: {esc(m['injury_signals_b'] or 'none')}{' · manual flag ' + str(m['injury_flag_b']) if m.get('injury_flag_b') else ''}")

with right:
    st.markdown("#### Prices")
    best = {"a": m.get("best_a"), "b": m.get("best_b")}
    st.caption(f"Best available: {a} {best['a'] or '—'} ({m.get('best_book_a') or '—'}) · {b} {best['b'] or '—'} ({m.get('best_book_b') or '—'}) "
               f"· Pinnacle {m.get('pinnacle_a') or '—'} / {m.get('pinnacle_b') or '—'} · {m.get('n_books', 0)} books")
    with st.container(border=True):
        st.markdown("**💱 Check your price**")
        x1, x2 = st.columns(2)
        side = x1.radio("Player", [a, b], key=f"ck_side_{choice}")
        p = m["p_fair_a"] if side == a else 1 - m["p_fair_a"]
        dflt = (m.get("best_a") if side == a else m.get("best_b")) or round(1 / p, 2)
        price = x2.number_input("Odds", 1.01, 100.0, float(dflt), 0.01, key=f"ck_odds_{choice}_{side}")
        e = p * price - 1
        stake = kelly_stake(p, price, settings.kelly_fraction, settings.kelly_cap)
        if e >= settings.min_ev:
            st.success(f"Value: EV {e:+.1%}. Stake {stake:.2%} of bankroll.")
        else:
            st.warning(f"No value: EV {e:+.1%}. Need at least {(1 + settings.min_ev) / p:.2f}.")
    sets = m.get("sets") or {}
    if sets.get("available"):
        st.markdown("#### Total sets 3.5")
        st.metric("P(4+ sets)", f"{sets['p_over35']:.0%}", help="See the Grand Slam sets page for method and calibration")
        sr = sets.get("recommendation")
        if sr:
            st.success(f"BET {sr['selection']} @ {sr['price']:.2f} · EV {sr['ev']:+.1%} · stake {sr['stake']:.2%}")
        st.caption(f"Fair over {sets['fair_over']:.2f} · fair under {sets['fair_under']:.2f} · best over {sets.get('best_over') or '—'} · best under {sets.get('best_under') or '—'}")
