import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import time  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (age_text, badge, bar, book_name, current_slate, esc, fmt_odds, keys_available, kpi, level_name,  # noqa: E402
                    local_time, money, price_age_minutes, runtime, settings_sidebar, setup)

setup("ATP Edge")
settings = settings_sidebar()
with st.sidebar:
    st.divider()
    if keys_available() and st.button("🔄 Fetch live prices now", use_container_width=True):
        st.session_state["live_nonce"] = int(time.time())
        st.session_state["demo"] = False
    st.session_state["demo"] = st.toggle("Demo data (made-up prices)", value=st.session_state.get("demo", False),
                                         help="Explore the app without API keys. Prices are invented: never bet from demo data.")

slate, label, demo = current_slate(settings)
rt = runtime()
price_first = settings.strategy == "blend"

st.markdown(
    '<div class="hero"><h1>🎾 ATP Edge</h1><div class="sub">What to bet on men\'s tennis today, how much, and the lowest '
    'price worth taking. ATP 250 / 500 / Masters 1000 / Grand Slams / ATP Finals.</div></div>'
    + badge(label, "ok" if "prices checked" in label or label.startswith("live") else "")
    + badge(f"Results in the model up to {rt.metrics.get('latest_match_date', '?')}", "")
    + badge(("Rule: price beats the sharp market by " if price_first else "Rule (old): model EV ≥ ") + f"{settings.min_ev:.0%}+", "blue")
    + (badge("DEMO · made-up prices · do not bet", "bad") if demo else ""),
    unsafe_allow_html=True,
)

matches = slate["matches"]
ml_bets = [m for m in matches if m.get("recommendation")]
set_bets = [m for m in matches if (m.get("sets") or {}).get("recommendation")]
cards = [(m, m["recommendation"], "ml") for m in ml_bets] + [(m, m["sets"]["recommendation"], "sets") for m in set_bets]
priced = sum(1 for m in matches if m.get("p_market_a") is not None)
bankroll = float(st.session_state.get("bankroll", 0) or 0)
total_stake = sum(r["stake"] for _, r, _ in cards)
ages = [a for a in (price_age_minutes(m.get("prices_ts")) for m in matches) if a is not None]

k = st.columns(4)
kpi(k[0], "Bets right now", len(cards), (f"total stake {money(total_stake)}" if cards else "nothing passes the rule"))
kpi(k[1], "Matches in the next 48 h", len(matches), f"{priced} with bookmaker prices")
kpi(k[2], "Prices last checked", (age_text(time.time() - min(ages) * 60) if ages else "—"),
    "re-check the price before you bet" if ages and min(ages) > 30 else "refreshed automatically")
try:
    from atp_edge.tracking import summary
    live = summary()
except Exception:
    live = {}
if live.get("settled_bets"):
    kpi(k[3], "Live record", f"{live['settled_bets']} bets settled",
        f"ROI {live['roi']:+.1%} · beat the closing price by {live['avg_clv']:+.1%}" if live.get("roi") is not None else "")
else:
    kpi(k[3], "Live record", "no settled bets yet", "see Results for the running score")

if demo:
    st.markdown('<div class="note"><b>Demo mode.</b> These prices are invented so you can see how the page works. '
                'Switch off “Demo data” in the sidebar to see real prices.</div>', unsafe_allow_html=True)

with st.expander("How to use this page", expanded=not st.session_state.get("seen_help", False) and not cards):
    st.session_state["seen_help"] = True
    st.markdown(
        '<div class="steps">'
        '<div class="step"><div class="n">Step 1</div><div class="t">Set up once</div><div class="d">In the sidebar, enter your '
        'bankroll and tick the sportsbooks you have accounts with. Stakes and best prices follow from those two things.</div></div>'
        '<div class="step"><div class="n">Step 2</div><div class="t">Look at the green cards</div><div class="d">Each card is one bet: '
        'who, how much, at which book, and the lowest price worth taking. No card, no bet. Most days have few or none.</div></div>'
        '<div class="step"><div class="n">Step 3</div><div class="t">Check the price is still there</div><div class="d">Prices move. '
        'If the book now offers less than the “don\'t take below” price, skip the bet. Never chase it.</div></div>'
        '<div class="step"><div class="n">Step 4</div><div class="t">Judge it over hundreds of bets</div><div class="d">Any single bet '
        'loses often. The Results page tracks whether you are beating the closing price, the earliest sign of a real edge.</div></div>'
        '</div>', unsafe_allow_html=True)
    st.caption("Why a bet appears: a bookmaker is offering a noticeably better price than the sharpest bookmaker's price with its "
               "margin removed, and the model doesn't disagree. See “How it works” for the reasoning and the test results.")

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


def open_match(m):
    st.session_state["match_detail"] = m
    st.switch_page("views/match.py")


# ---------------------------------------------------------------- bets
st.markdown("### ✅ Bets to place now")
if not cards:
    st.markdown('<div class="empty"><b>No bets right now.</b> No bookmaker is offering a price that beats the fair price by '
                f'{settings.min_ev:.0%} or more. That is normal: passing is part of the method. The table below shows the '
                'price each player would need.</div>', unsafe_allow_html=True)
else:
    cols = st.columns(3 if len(cards) > 2 else max(len(cards), 2))
    for i, (m, r, kind) in enumerate(sorted(cards, key=lambda c: -c[1]["ev"])):
        age = price_age_minutes(m.get("prices_ts"))
        warns = []
        if age is not None and age > 30:
            warns.append(f"Price seen {age_text(m.get('prices_ts'))}: check it before betting.")
        nb = r.get("next_best_price")
        if nb and r["price"] / nb - 1 > 0.06:
            warns.append(f"Much higher than the next book ({fmt_odds(nb)}): may be gone quickly, or a mistake the book can cancel.")
        min_price = r.get("min_price")
        floor = f'<div><span>Don\'t take below</span><b>{fmt_odds(min_price)}</b></div>' if min_price else ""
        with cols[i % len(cols)]:
            st.markdown(
                f'<div class="bet {"sets" if kind == "sets" else ""}">'
                f'<div class="act">Bet <b>{esc(money(r["stake"]))}</b> on</div>'
                f'<div class="p">{esc(r["selection"])} @ {fmt_odds(r["price"])}</div>'
                f'<div class="m">at <b>{esc(book_name(r.get("book")))}</b> · {esc(m["player_a"])} vs {esc(m["player_b"])}<br>'
                f'{esc(m["event"])} · starts {esc(local_time(m.get("start_ts")))}</div><div class="n">'
                f'<div><span>Value</span><b>{r["ev"]:+.1%}</b></div>'
                f'<div><span>Chance to win</span><b>{r["probability"]:.0%}</b></div>'
                f'<div><span>Fair price</span><b>{fmt_odds(r["fair_odds"])}</b></div>{floor}</div>'
                + "".join(f'<div class="warn">⚠️ {esc(w)}</div>' for w in warns) + '</div>', unsafe_allow_html=True)
            if st.button("Why this bet →", key=f"bet_{kind}_{i}", use_container_width=True):
                open_match(m)
    st.caption("“Value” is the expected profit per dollar staked if the fair price is right. A 60% chance still loses 4 times in 10: "
               "stakes are kept small on purpose.")

# ---------------------------------------------------------------- all matches
st.markdown("### 📋 All matches")
st.caption("Fair chance = the sharp market price with the bookmaker's margin removed, nudged by the model. "
           "“Needs” is the lowest price that would make each player a bet." if price_first else
           "Fair chance = the model's own probability (old rule).")
df = pd.DataFrame(matches)
W = [0.8, 3.2, 1.25, 1.25, 2.3, 0.8]
for t_i, (event, grp) in enumerate(df.groupby("event", sort=False)):
    first = grp.iloc[0]
    n_b = int(grp["recommendation"].notna().sum())
    head = (f"{event} · {level_name(first['level'])} · {first['surface']}{' (indoor)' if first.get('indoor') else ''} · "
            f"{len(grp)} matches" + (f" · ✅ {n_b} bet{'s' if n_b > 1 else ''}" if n_b else ""))
    with st.expander(head, expanded=t_i < 3):
        h = st.columns(W)
        for c, t in zip(h, ["Starts", "Match · fair chance to win", "Best price", "Needs", "Verdict", ""]):
            c.markdown(f'<span class="muted">{t}</span>', unsafe_allow_html=True)
        for j, m in grp.iterrows():
            c = st.columns(W, vertical_alignment="center")
            pa = float(m["p_fair_a"])
            c[0].markdown(f"**{esc(local_time(m.get('start_ts')))}**")
            inj = " 🩹" if (m.get("injury_signals_a") or m.get("injury_flag_a")) else ""
            injb = " 🩹" if (m.get("injury_signals_b") or m.get("injury_flag_b")) else ""
            c[1].markdown(f'**{esc(m["player_a"])}**{inj} <span class="muted">{pa:.0%}</span> vs **{esc(m["player_b"])}**{injb} '
                          f'<span class="muted">{1 - pa:.0%}</span>{bar(pa)}', unsafe_allow_html=True)
            ba, bb = m.get("best_a"), m.get("best_b")
            has_price = pd.notna(ba) and pd.notna(bb) and bool(ba) and bool(bb)
            age = age_text(m.get("prices_ts")) if pd.notna(m.get("prices_ts")) else ""
            c[2].markdown((f"{fmt_odds(ba)} / {fmt_odds(bb)}" + (f'<br><span class="muted">{esc(age)}</span>' if age else ""))
                          if has_price else '<span class="muted">no price yet</span>', unsafe_allow_html=True)
            va = m.get("verdict_a") if isinstance(m.get("verdict_a"), dict) else {}
            vb = m.get("verdict_b") if isinstance(m.get("verdict_b"), dict) else {}
            na, nb_ = va.get("min_price"), vb.get("min_price")
            c[3].markdown(f"{fmt_odds(na)} / {fmt_odds(nb_)}" if na and nb_ else f"{fmt_odds(m['fair_odds_a'])} / {fmt_odds(m['fair_odds_b'])}"
                          '<br><span class="muted">fair price</span>', unsafe_allow_html=True)
            r = m.get("recommendation")
            sets = m.get("sets") if isinstance(m.get("sets"), dict) else {}
            chips = ""
            if isinstance(r, dict):
                chips += badge(f"BET {r['selection'].split()[-1]} {r['ev']:+.1%}", "ok")
            elif has_price and m.get("market_reference_ok") is False and price_first:
                chips += badge("Too few books to judge", "warn")
            elif has_price:
                chips += badge("No value at these prices", "")
            else:
                fav = m["player_a"] if pa >= 0.5 else m["player_b"]
                chips += badge(f"Model leans {fav.split()[-1]}", "blue")
            if sets.get("available"):
                sr = sets.get("recommendation")
                chips += badge(f"{sr['selection']} {sr['ev']:+.1%}", "ok") if sr else badge(f"4+ sets {sets.get('p_fair_over', sets['p_over35']):.0%}", "blue")
            c[4].markdown(chips, unsafe_allow_html=True)
            if c[5].button("Open", key=f"open_{t_i}_{j}"):
                open_match(m.to_dict())
st.caption("🩹 = recent retirement, walkover, long layoff or a manual injury flag. Prices show as first player / second player.")

if slate.get("skipped"):
    with st.expander(f"{len(slate['skipped'])} fixtures not shown"):
        st.dataframe(pd.DataFrame(slate["skipped"])[["player_a", "player_b", "tournament", "reason"]], hide_index=True, use_container_width=True)
with st.expander("Data sources (technical)"):
    st.json(slate.get("sources", {}))
