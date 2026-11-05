import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import badge, current_slate, esc, kpi, local_time, runtime, settings_sidebar, setup  # noqa: E402

setup("Grand Slam sets")
settings = settings_sidebar()
slate, label, demo = current_slate(settings)
sm = runtime().metrics.get("sets_model", {})

st.markdown('<div class="hero"><h1>🏆 Grand Slam · total sets 3.5</h1><div class="sub">Men\'s best-of-five matches at '
            'the Australian Open, Roland Garros, Wimbledon and the US Open. Over 3.5 = the match goes to at least '
            'four sets.</div></div>' + badge(f"Data: {label}") + (badge("DEMO · synthetic prices", "bad") if demo else ""),
            unsafe_allow_html=True)

if sm.get("available"):
    k = st.columns(4)
    kpi(k[0], "Model log loss", f"{sm['log_loss']:.3f}", f"walk-forward, {sm['eval_matches']:,} Slam matches")
    kpi(k[1], "Base-rate log loss", f"{sm['base_rate_log_loss']:.3f}", f"always predicting {sm['over_rate']:.0%}")
    kpi(k[2], "Naive set-by-set", f"{sm['independent_sets_log_loss']:.3f}", "treats sets as independent")
    kpi(k[3], "Over 3.5 rate", f"{sm['over_rate']:.0%}", f"{sm['matches']:,} Slam matches since 2000")
    with st.expander("How the over/under 3.5 sets probability is built"):
        st.markdown(
            "1. **Who is favoured, and by how much.** The Pinnacle moneyline (no-vig) when available, otherwise the "
            "rating model.\n"
            "2. **Per-set strength.** That match probability is converted to the chance of winning a single set.\n"
            "3. **Naive four-plus-set chance.** Treating sets as independent gives a first estimate. On its own it "
            "over-predicts long matches, because favourites run away with matches more often than independence implies.\n"
            "4. **Learned correction (logistic regression).** It adjusts for: both players' serve strength (big servers mean "
            "tighter sets and more tiebreaks); serve-vs-return mismatch; the court speed index; each player's history "
            "of long Slam matches; best-of-five experience; recent injury signals; and round.\n\n"
            + sm.get("note", ""))
        cal = pd.DataFrame(sm.get("calibration", []))
        if not cal.empty:
            st.caption("Calibration on walk-forward seasons (predicted vs actual over-3.5 rate)")
            st.dataframe(cal, hide_index=True, use_container_width=True,
                         column_config={"predicted": st.column_config.NumberColumn(format="percent"),
                                        "actual": st.column_config.NumberColumn(format="percent")})

rows = [m for m in slate["matches"] if (m.get("sets") or {}).get("available")]
if not rows:
    st.markdown('<div class="empty">No Grand Slam best-of-five matches in the next 48 hours. This page fills '
                'automatically during the Australian Open, Roland Garros, Wimbledon and the US Open.</div>',
                unsafe_allow_html=True)
    st.stop()

st.markdown("### Matches")
for i, m in enumerate(rows):
    s = m["sets"]
    rec = s.get("recommendation")
    with st.container(border=True):
        c = st.columns([2.6, 1.2, 1.2, 1.3, 2.2], vertical_alignment="center")
        c[0].markdown(f"**{esc(m['player_a'])}** vs **{esc(m['player_b'])}**<br><span class='muted'>{esc(m['event'])} · "
                      f"{esc(local_time(m.get('start_ts')))} · fair {m['p_fair_a']:.0%} / {1 - m['p_fair_a']:.0%} · "
                      f"court speed {m['court_speed']:.2f}</span>", unsafe_allow_html=True)
        c[1].metric("P(over 3.5)", f"{s['p_over35']:.0%}")
        bo, bu = s.get("best_over"), s.get("best_under")
        c[2].markdown(f"<span class='muted'>Fair over / under</span><br><b style='font-size:1.25rem'>{s['fair_over']:.2f} / {s['fair_under']:.2f}</b>",
                      unsafe_allow_html=True)
        c[3].markdown(f"<span class='muted'>Best price O / U</span><br><b style='font-size:1.25rem'>"
                      + (f"{bo:.2f} / {bu:.2f}" if bo and bu else "—") + "</b>", unsafe_allow_html=True)
        if rec:
            c[4].markdown(badge(f"BET {rec['selection']} @ {rec['price']:.2f}", "ok") + "<br>" +
                          f"<span class='muted'>EV {rec['ev']:+.1%} · stake {rec['stake']:.2%} · {esc(rec.get('book') or '')}</span>",
                          unsafe_allow_html=True)
        else:
            c[4].markdown(badge("No value" if bo else "No sets price yet", ""), unsafe_allow_html=True)
        with st.expander("Price checker & set-score view"):
            p1, p2 = st.columns(2)
            side = p1.radio("Side", ["Over 3.5", "Under 3.5"], horizontal=True, key=f"s_side_{i}")
            p = s["p_over35"] if side.startswith("Over") else 1 - s["p_over35"]
            price = p2.number_input("Your odds", 1.01, 50.0, float(round(1 / p, 2)), 0.01, key=f"s_odds_{i}_{side}")
            e = p * price - 1
            (st.success if e >= settings.min_ev_sets else st.info)(f"EV {e:+.1%} at {price:.2f} (fair {1 / p:.2f}).")
            dist = pd.DataFrame([{"Score": k.replace("A", m["player_a"]).replace("B", m["player_b"]), "Chance (independent sets)": v}
                                 for k, v in s["independent_sets_view"].items()])
            st.caption("Set-score breakdown assuming independent sets (before the learned correction):")
            st.dataframe(dist, hide_index=True, use_container_width=True,
                         column_config={"Chance (independent sets)": st.column_config.NumberColumn(format="percent")})
