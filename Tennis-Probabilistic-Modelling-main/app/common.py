"""Shared state and styling for the Streamlit pages."""
from __future__ import annotations

import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_edge.config import APP_TIMEZONE, secret  # noqa: E402
from atp_edge.predict import LATEST_PATH, STRATEGIES, Runtime, Settings, build_slate  # noqa: E402
from atp_edge.tournaments import level_label  # noqa: E402

TZ = ZoneInfo(APP_TIMEZONE)

CSS = """
<style>
:root{--ok:#3ddc84;--warn:#fab219;--bad:#e66767;--muted:#8b949e;--card:#161b22;--line:#2a313c;--a:#3987e5;--b:#d95926;}
.block-container{padding-top:2rem;max-width:1300px}
.hero h1{font-size:2rem;margin:0}.hero .sub{color:var(--muted);margin:.2rem 0 .6rem}
.badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.76rem;font-weight:600;border:1px solid var(--line);
 background:#1c2230;color:#c9d1d9;margin:0 .3rem .3rem 0;white-space:nowrap}
.badge.ok{color:var(--ok);border-color:rgba(61,220,132,.4);background:rgba(61,220,132,.1)}
.badge.warn{color:var(--warn);border-color:rgba(250,178,25,.4);background:rgba(250,178,25,.1)}
.badge.bad{color:var(--bad);border-color:rgba(230,103,103,.4);background:rgba(230,103,103,.1)}
.badge.blue{color:#79b0f2;border-color:rgba(57,135,229,.4);background:rgba(57,135,229,.1)}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.75rem 1rem;height:100%}
.kpi .l{color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{font-size:1.5rem;font-weight:700;margin-top:.1rem}.kpi .h{color:var(--muted);font-size:.8rem}
.bet{border:1px solid rgba(61,220,132,.4);background:linear-gradient(180deg,rgba(61,220,132,.09),rgba(61,220,132,.02));
 border-radius:14px;padding:.85rem 1rem;margin-bottom:.4rem}
.bet.sets{border-color:rgba(57,135,229,.45);background:linear-gradient(180deg,rgba(57,135,229,.1),rgba(57,135,229,.02))}
.bet .p{font-size:1.15rem;font-weight:700}.bet .m{color:var(--muted);font-size:.84rem;margin-top:.1rem}
.bet .n{display:flex;gap:1.3rem;flex-wrap:wrap;margin-top:.5rem}
.bet .n span{display:block;color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.05em}
.bar{height:8px;border-radius:999px;background:#2a313c;overflow:hidden;display:flex;margin:.25rem 0}
.bar .a{background:var(--a)}.bar .g{width:2px;background:var(--card)}.bar .b{background:var(--b);flex:1}
.bet .act{font-size:1.02rem;margin:.1rem 0 .15rem}.bet .act b{color:var(--ok)}
.bet.sets .act b{color:#79b0f2}
.bet .warn{color:var(--warn);font-size:.8rem;margin-top:.45rem}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.6rem;margin:.3rem 0 .8rem}
.step{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem}
.step .n{color:var(--ok);font-weight:700;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase}
.step .t{font-weight:700;margin:.1rem 0 .15rem}.step .d{color:var(--muted);font-size:.84rem;line-height:1.35}
.note{border-left:3px solid var(--warn);background:rgba(250,178,25,.07);padding:.55rem .8rem;border-radius:0 8px 8px 0;
 font-size:.88rem;margin:.4rem 0 .8rem}
.note.ok{border-left-color:var(--ok);background:rgba(61,220,132,.07)}
.muted{color:var(--muted);font-size:.84rem}
.empty{border:1px dashed var(--line);border-radius:14px;padding:1rem 1.2rem;background:var(--card)}
</style>
"""


def setup(title: str, icon: str = "🎾") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    auto_refresh()


def esc(x) -> str:
    return html.escape(str(x))


def badge(text, kind="") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def kpi(col, label, value, hint=""):
    col.markdown(f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                 f'<div class="h">{esc(hint)}</div></div>', unsafe_allow_html=True)


def bar(p_a: float) -> str:
    p = max(0.0, min(1.0, float(p_a)))
    return f'<div class="bar"><div class="a" style="width:{p*100:.1f}%"></div><div class="g"></div><div class="b"></div></div>'


def local_time(ts) -> str:
    if not ts:
        return "TBD"
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(TZ).strftime("%a %H:%M")


def _artifact_stamp() -> tuple:
    from atp_edge.modeling import ENGINE_PATH, MATCH_MODEL
    return tuple(int(x.stat().st_mtime) if x.exists() else 0 for x in (ENGINE_PATH, MATCH_MODEL, LATEST_PATH))


@st.cache_resource
def _runtime(stamp: tuple) -> Runtime:
    return Runtime()


def runtime() -> Runtime:
    """Reloads automatically when the daily job commits a retrained model."""
    return _runtime(_artifact_stamp()[:2])


@st.fragment(run_every="2m")
def auto_refresh() -> None:
    """Re-run the page when the live-prices job has published new prices (checked every 2 min)."""
    stamp = _artifact_stamp()
    if st.session_state.get("_stamp") is None:
        st.session_state["_stamp"] = stamp
    elif st.session_state["_stamp"] != stamp:
        st.session_state["_stamp"] = stamp
        st.rerun(scope="app")


def _books_seen() -> list[str]:
    try:
        f, o, *_ = load_inputs()
    except Exception:
        return []
    seen = set()
    for ev in list(f) + list(o):
        seen.update((ev.get("prices_a") or {}).keys())
    return sorted(seen)


RISK_LEVELS = {
    # label: (Kelly fraction, max stake per bet, one-line description)
    "Cautious": (0.125, 0.01, "⅛-Kelly, at most 1% of bankroll per bet"),
    "Standard": (0.25, 0.02, "¼-Kelly, at most 2% of bankroll per bet"),
    "Bold": (0.5, 0.03, "½-Kelly, at most 3% per bet: bigger swings"),
}


def _keep(name: str, default):
    """A sidebar value that survives moving between pages (widget state alone does not)."""
    return st.session_state.setdefault("cfg_" + name, default)


def _save(name: str, value):
    st.session_state["cfg_" + name] = value
    return value


def fmt_odds(x) -> str:
    """Decimal odds, or American when chosen in the sidebar."""
    try:
        o = float(x)
    except (TypeError, ValueError):
        return "—"
    if not o or o <= 1:
        return "—"
    if st.session_state.get("cfg_odds_format", "Decimal") == "American":
        return f"+{(o - 1) * 100:.0f}" if o >= 2 else f"−{100 / (o - 1):.0f}"
    return f"{o:.2f}"


def money(fraction) -> str:
    """A stake given as a share of bankroll, in dollars when a bankroll is set."""
    bankroll = float(st.session_state.get("bankroll", 0) or 0)
    try:
        f = float(fraction)
    except (TypeError, ValueError):
        return "—"
    return f"CA${bankroll * f:,.2f}" if bankroll else f"{f:.2%} of bankroll"


def book_name(key) -> str:
    known = {"pinnacle": "Pinnacle", "bet365": "bet365", "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
             "betrivers": "BetRivers", "williamhill": "William Hill", "williamhill_us": "Caesars", "unibet": "Unibet",
             "betway": "Betway", "bwin": "bwin", "marathonbet": "Marathonbet", "sport888": "888sport", "betfair_ex_eu": "Betfair Exchange",
             "betonlineag": "BetOnline", "bovada": "Bovada", "pointsbetus": "PointsBet", "espnbet": "ESPN BET", "fonbet": "Fonbet"}
    k = str(key or "")
    return known.get(k, k.replace("_", " ").title() if k else "—")


def settings_sidebar() -> Settings:
    """Sidebar controls. Defaults are the same rule the automated bet log uses."""
    d = Settings()
    with st.sidebar:
        st.markdown("### 💰 Your setup")
        bankroll = st.number_input("Bankroll (CA$)", min_value=0.0, value=float(st.session_state.get("bankroll", 1000.0)), step=100.0,
                                   help="Money set aside only for betting. Every stake is a small share of this number.")
        st.session_state["bankroll"] = bankroll
        seen = _books_seen()
        saved = _keep("books", [b for b in d.books if b in seen] if d.books else [])
        books = st.multiselect("Sportsbooks you can use", seen, default=[b for b in saved if b in seen], format_func=book_name,
                               help="Only these are searched for the best price. Leave empty to search every book in the feed "
                                    "(you will then see prices at books you may not have an account with).")
        _save("books", books)
        if seen and not books:
            st.caption("⚠️ No books picked: bets may point to books you can't use.")
        levels = list(RISK_LEVELS)
        default_level = next((k for k, v in RISK_LEVELS.items() if abs(v[0] - d.kelly_fraction) < 1e-9 and abs(v[1] - d.kelly_cap) < 1e-9), "Custom")
        level = st.radio("Stake size", levels + ["Custom"], index=(levels + ["Custom"]).index(_keep("risk", default_level)), horizontal=True,
                         help="How much of the mathematically 'optimal' (Kelly) stake to bet. Smaller = smoother ride.")
        _save("risk", level)
        if level == "Custom":
            fr_opts = [0.1, 0.125, 0.2, 0.25, 0.33, 0.5]
            cur = _keep("kelly_fraction", d.kelly_fraction if d.kelly_fraction in fr_opts else 0.25)
            frac = _save("kelly_fraction", st.select_slider("Kelly fraction", fr_opts, cur, format_func=lambda x: {
                0.1: "1/10", 0.125: "1/8", 0.2: "1/5", 0.25: "1/4", 0.33: "1/3", 0.5: "1/2"}[x]))
            cap = _save("kelly_cap", st.slider("Max stake per bet", 0.5, 5.0, float(_keep("kelly_cap", d.kelly_cap)) * 100, 0.5, format="%.1f%%") / 100)
        else:
            frac, cap = RISK_LEVELS[level][0], RISK_LEVELS[level][1]
            st.caption(RISK_LEVELS[level][2])
        min_ev = _save("min_ev", st.slider("Minimum value to bet", 1.0, 12.0, float(_keep("min_ev", d.min_ev)) * 100, 0.5, format="%.1f%%",
                                           help="How much better than the fair price a book's price must be. Higher = fewer, "
                                                "stronger bets. 3% is the tested default.") / 100)
        with st.expander("Advanced"):
            keys = list(STRATEGIES)
            strategy = _save("strategy", st.radio("How the fair price is set", keys, index=keys.index(_keep("strategy", d.strategy)),
                                                  format_func=lambda k: {"blend": "Price first (recommended)", "model": "Model only (old rule)"}[k],
                                                  help="Price first: starts from the sharpest bookmaker's price and lets the model nudge it. "
                                                       "Model only: trusts the model over the market; in testing this lost money at closing prices."))
            max_odds = _save("max_odds", st.slider("Skip outsiders above odds of", 3.0, 30.0, float(_keep("max_odds", d.max_odds)), 0.5))
            min_edge = d.min_edge
            if strategy == "model":
                min_edge = _save("min_edge", st.slider("Minimum edge vs market", 0.0, 20.0, float(_keep("min_edge", d.min_edge)) * 100, 0.5, format="%.1f%%") / 100)
            min_ev_sets = _save("min_ev_sets", st.slider("Minimum value (Slam sets)", 0.0, 20.0, float(_keep("min_ev_sets", d.min_ev_sets)) * 100, 0.5, format="%.1f%%") / 100)
            fmt_opts = ["Decimal", "American"]
            _save("odds_format", st.radio("Odds format", fmt_opts, index=fmt_opts.index(_keep("odds_format", "Decimal")), horizontal=True))
            st.caption("The automated bet log and phone alerts use the repo defaults (variables STRATEGY, MIN_EV, MAX_ODDS, "
                       "KELLY_FRACTION, KELLY_CAP, BET_BOOKS), not these sliders.")
        return Settings(strategy=strategy, min_ev=min_ev, min_edge=min_edge, min_ev_sets=min_ev_sets,
                        kelly_fraction=frac, kelly_cap=cap, max_odds=max_odds, books=tuple(books))


def price_age_minutes(ts) -> float | None:
    try:
        return (time.time() - float(ts)) / 60
    except (TypeError, ValueError):
        return None


def keys_available() -> bool:
    return bool(secret("RAPIDAPI_KEY") or secret("ODDS_API_KEY"))


@st.cache_data(ttl=600, show_spinner="Fetching live fixtures and prices…")
def live_inputs(_nonce: int):
    from atp_edge.live import collect
    fixtures, odds_events, status = collect(runtime(), demo=False, persist=False)
    return fixtures, odds_events, status, time.time()


@st.cache_data(ttl=3600)
def demo_inputs(_day: str):
    from atp_edge.live import collect
    fixtures, odds_events, status = collect(runtime(), demo=True)
    return fixtures, odds_events, status


def load_inputs() -> tuple[list, list, dict, str, bool]:
    """Returns fixtures, odds events, source status, a freshness label, demo flag."""
    if st.session_state.get("demo"):
        f, o, s = demo_inputs(time.strftime("%Y-%m-%d"))
        return f, o, s, "demo data", True
    if st.session_state.get("live_nonce") and keys_available():
        f, o, s, ts = live_inputs(st.session_state["live_nonce"])
        return f, o, s, "live " + datetime.fromtimestamp(ts, tz=TZ).strftime("%H:%M"), False
    if LATEST_PATH.exists():
        data = json.loads(LATEST_PATH.read_text())
        inp = data.get("inputs") or {}
        ts = (data.get("sources") or {}).get("refreshed_ts")
        label = ("prices checked " + datetime.fromtimestamp(float(ts), tz=TZ).strftime("%a %H:%M")) if ts else \
            "automated run " + str(data.get("generated_at_local", ""))
        return inp.get("fixtures", []), inp.get("odds_events", []), data.get("sources", {}), label, bool(data.get("demo"))
    return [], [], {}, "no data yet", False


def current_slate(settings: Settings) -> tuple[dict, str, bool]:
    f, o, status, label, demo = load_inputs()
    slate = build_slate(runtime(), f, o, settings)
    slate["sources"] = status
    return slate, label, demo


def age_text(ts) -> str:
    """'4 min ago' for a price timestamp."""
    try:
        mins = (time.time() - float(ts)) / 60
    except (TypeError, ValueError):
        return ""
    if mins < 1:
        return "just now"
    return f"{mins:.0f} min ago" if mins < 90 else f"{mins / 60:.0f} h ago"


def level_name(code) -> str:
    return level_label(code)
