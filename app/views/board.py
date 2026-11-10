import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (age_text, badge, bar, current_slate, esc, keys_available, kpi, level_name, local_time,  # noqa: E402
                    runtime, settings_sidebar, setup)

setup("ATP Edge")
settings = settings_sidebar()
with st.sidebar:
    st.divider()
    if keys_available() and st.button("🔄 Fetch live prices now", use_container_width=True):
        st.session_state["live_nonce"] = int(pd.Timestamp.now().timestamp())
        st.session_state["demo"] = False
    st.session_state["demo"] = st.toggle("Demo data (synthetic prices)", value=st.session_state.get("demo", False))

slate, label, demo = current_slate(settings)
rt = runtime()
bench = rt.metrics.get("market_benchmark", {})

st.markdown(
    '<div class="hero"><h1>🎾 ATP Edge</h1><div class="sub">Men\'s singles · ATP 250 / 500 / Masters 1000 / '
    'Grand Slams / ATP Finals · fair odds, value bets and Grand Slam total sets</div></div>'
    + badge(f"{label} · auto-refresh every 10 min", "ok" if "prices checked" in label else "")
    + badge(f"Ratings to {rt.metrics.get('latest_match_date', '?')}", "")
    + badge(f"Bets: EV ≥ {settings.min_ev:.0%} · edge ≥ {settings.min_edge:.0%}", "blue")
    + (badge("DEMO · synthetic prices", "bad") if demo else ""),
    unsafe_allow_html=True,
)

matches = slate["matches"]
ml_bets = [m for m in matches if m.get("recommendation")]
set_bets = [m for m in matches if (m.get("sets") or {}).get("recommendation")]
priced = sum(1 for m in matches if m.get("p_market_a") is not None)
bankroll = float(st.session_state.get("bankroll", 0) or 0)

k = st.columns(4)
kpi(k[0], "Matches (next 48 h)", len(matches), f"{priced} with prices")
kpi(k[1], "Moneyline value bets", len(ml_bets), "price beats fair odds" if ml_bets else "none right now")
kpi(k[2], "Slam over/under 3.5 sets", len(set_bets), "Grand Slam best-of-5 only")
if bench:
    kpi(k[3], "Model vs market", f"{bench['model_log_loss']:.3f} / {bench['pinnacle_log_loss']:.3f}", "log loss 2018-26 (lower wins)")

if not matches:
    srcs = slate.get("sources", {})
    errs = "; ".join(f"{k}: {v.get('error')}" for k, v in srcs.items() if isinstance(v, dict) and v.get("error"))
    st.markdown(
        '<div class="empty"><b>No upcoming tour-level matches in the next 48 hours.</b><br>'
        + (f"Source messages: {esc(errs)}<br>" if errs else "")
        + ("Add RAPIDAPI_KEY and ODDS_API_KEY to the app secrets, or switch on <b>Demo data</b> in the sidebar to explore."
           if not keys_available() else "Between tournaments the board is empty; it fills as soon as draws are published.")
        + "</div>", unsafe_allow_html=True)
    if slate.get("skipped"):
        with st.expander(f"{len(slate['skipped'])} matches skipped"):
            st.dataframe(pd.DataFrame(slate["skipped"])[["player_a", "player_b", "tournament", "reason"]], hide_index=True)
    st.stop()


def stake_text(frac: float) -> str:
    return f"{frac:.2%}" + (f" · CA${bankroll * frac:,.0f}" if bankroll else "")


def open_match(m):
    st.session_state["match_detail"] = m
    st.switch_page("views/match.py")


# ---------------------------------------------------------------- bets
st.markdown("### ✅ Recommended bets")
cards = [(m, m["recommendation"], "ml") for m in ml_bets] + [(m, m["sets"]["recommendation"], "sets") for m in set_bets]
if not cards:
    st.markdown('<div class="empty">No match passes your EV and edge thresholds right now. Open any match to see the '
                'model\'s probability against every book\'s price, or type in your own price.</div>', unsafe_allow_html=True)
else:
    cols = st.columns(3)
    for i, (m, r, kind) in enumerate(sorted(cards, key=lambda c: -c[1]["ev"])):
        with cols[i % 3]:
            st.markdown(
                f'<div class="bet {"sets" if kind == "sets" else ""}"><div class="p">{esc(r["selection"])} @ {r["price"]:.2f}</div>'
                f'<div class="m">{esc(m["player_a"])} vs {esc(m["player_b"])}<br>{esc(m["event"])} · {esc(local_time(m.get("start_ts")))}'
                f' · {esc(r.get("book") or "")}</div><div class="n">'
                f'<div><span>EV</span><b>{r["ev"]:+.1%}</b></div><div><span>Fair odds</span><b>{r["fair_odds"]:.2f}</b></div>'
                f'<div><span>Win prob</span><b>{r["probability"]:.0%}</b></div><div><span>Stake</span><b>{esc(stake_text(r["stake"]))}</b></div>'
                f'</div></div>', unsafe_allow_html=True)
            if st.button("Details →", key=f"bet_{kind}_{i}", use_container_width=True):
                open_match(m)

# ---------------------------------------------------------------- all matches
st.markdown("### 📋 All matches")
df = pd.DataFrame(matches)
for t_i, (event, grp) in enumerate(df.groupby("event", sort=False)):
    first = grp.iloc[0]
    n_b = int(grp["recommendation"].notna().sum())
    head = (f"{event} · {level_name(first['level'])} · {first['surface']}{' (indoor)' if first.get('indoor') else ''} · "
            f"court speed {first['court_speed']:.2f} · {len(grp)} matches" + (f" · ✅ {n_b}" if n_b else ""))
    with st.expander(head, expanded=t_i < 3):
        h = st.columns([0.8, 3.3, 1.2, 1.2, 2.4, 0.8])
        for c, t in zip(h, ["Time", "Match · fair win chance", "Best price", "Fair odds", "Verdict", ""]):
            c.markdown(f'<span class="muted">{t}</span>', unsafe_allow_html=True)
        for j, m in grp.iterrows():
            c = st.columns([0.8, 3.3, 1.2, 1.2, 2.4, 0.8], vertical_alignment="center")
            pa = float(m["p_fair_a"])
            c[0].markdown(f"**{esc(local_time(m.get('start_ts')))}**")
            inj = " 🩹" if (m.get("injury_signals_a") or m.get("injury_flag_a")) else ""
            injb = " 🩹" if (m.get("injury_signals_b") or m.get("injury_flag_b")) else ""
            c[1].markdown(f'**{esc(m["player_a"])}**{inj} <span class="muted">{pa:.0%}</span> vs **{esc(m["player_b"])}**{injb} '
                          f'<span class="muted">{1 - pa:.0%}</span>{bar(pa)}', unsafe_allow_html=True)
            ba, bb = m.get("best_a"), m.get("best_b")
            age = age_text(m.get("prices_ts")) if pd.notna(m.get("prices_ts")) else ""
            c[2].markdown((f"{ba:.2f} / {bb:.2f}" + (f'<br><span class="muted">{esc(age)}</span>' if age else ""))
                          if pd.notna(ba) and pd.notna(bb) and ba and bb else '<span class="muted">no price</span>', unsafe_allow_html=True)
            c[3].markdown(f"{m['fair_odds_a']:.2f} / {m['fair_odds_b']:.2f}")
            r = m.get("recommendation")
            sets = m.get("sets") if isinstance(m.get("sets"), dict) else {}
            chips = ""
            if isinstance(r, dict):
                chips += badge(f"BET {r['selection']} {r['ev']:+.1%}", "ok")
            elif pd.notna(ba) and ba:
                chips += badge("No value", "")
            else:
                fav = m["player_a"] if pa >= 0.5 else m["player_b"]
                chips += badge(f"Lean {fav}", "blue")
            if sets.get("available"):
                sr = sets.get("recommendation")
                chips += badge(f"{sr['selection']} {sr['ev']:+.1%}", "ok") if sr else badge(f"O3.5 sets {sets['p_over35']:.0%}", "blue")
            c[4].markdown(chips, unsafe_allow_html=True)
            if c[5].button("Open", key=f"open_{t_i}_{j}"):
                open_match(m.to_dict())

if slate.get("skipped"):
    with st.expander(f"{len(slate['skipped'])} fixtures not shown"):
        st.dataframe(pd.DataFrame(slate["skipped"])[["player_a", "player_b", "tournament", "reason"]], hide_index=True, use_container_width=True)
with st.expander("Data sources"):
    st.json(slate.get("sources", {}))
