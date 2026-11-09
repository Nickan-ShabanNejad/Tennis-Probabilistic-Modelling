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
from atp_edge.predict import LATEST_PATH, Runtime, Settings, build_slate  # noqa: E402
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


@st.fragment(run_every="10m")
def auto_refresh() -> None:
    """Re-run the page when the hourly job has published a new slate (checked every 10 min)."""
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


def settings_sidebar() -> Settings:
    """Sidebar controls. Defaults are the same rule the automated bet log uses."""
    d = Settings()
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        min_ev = st.slider("Minimum EV", 0.0, 20.0, d.min_ev * 100, 0.5, format="%.1f%%",
                           help="Expected return per unit staked at the best price, using the model's probability.") / 100
        min_edge = st.slider("Minimum edge vs market", 0.0, 20.0, d.min_edge * 100, 0.5, format="%.1f%%",
                             help="Model probability minus the market's no-vig probability (Pinnacle, else consensus).") / 100
        min_ev_sets = st.slider("Minimum EV (Slam sets)", 0.0, 20.0, d.min_ev_sets * 100, 0.5, format="%.1f%%") / 100
        frac = st.select_slider("Kelly fraction", [0.1, 0.2, 0.25, 0.33, 0.5], d.kelly_fraction if d.kelly_fraction in (0.1, 0.2, 0.25, 0.33, 0.5) else 0.25,
                                format_func=lambda x: {0.1: "1/10", 0.2: "1/5", 0.25: "1/4", 0.33: "1/3", 0.5: "1/2"}[x])
        cap = st.slider("Max stake per bet", 0.5, 5.0, d.kelly_cap * 100, 0.5, format="%.1f%%") / 100
        seen = _books_seen()
        default_books = [b for b in d.books if b in seen] if d.books else []
        books = st.multiselect("My sportsbooks", seen, default=default_books,
                               help="Only these books are used for the best price. Leave empty to use every book.")
        bankroll = st.number_input("Bankroll (CA$)", min_value=0.0, value=float(st.session_state.get("bankroll", 1000.0)), step=100.0)
        st.session_state["bankroll"] = bankroll
        st.caption("The automated bet log uses the defaults above (change them with repo variables MIN_EV, MIN_EDGE, "
                   "KELLY_FRACTION, KELLY_CAP, BET_BOOKS).")
        return Settings(strategy="model", min_ev=min_ev, min_edge=min_edge, min_ev_sets=min_ev_sets,
                        kelly_fraction=frac, kelly_cap=cap, books=tuple(books))


def keys_available() -> bool:
    return bool(secret("RAPIDAPI_KEY") or secret("ODDS_API_KEY"))


@st.cache_data(ttl=600, show_spinner="Fetching live fixtures and prices…")
def live_inputs(_nonce: int):
    from atp_edge.live import collect
    fixtures, odds_events, status = collect(runtime(), demo=False)
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
        label = "automated run " + str(data.get("generated_at_local", ""))
        return inp.get("fixtures", []), inp.get("odds_events", []), data.get("sources", {}), label, bool(data.get("demo"))
    return [], [], {}, "no data yet", False


def current_slate(settings: Settings) -> tuple[dict, str, bool]:
    f, o, status, label, demo = load_inputs()
    slate = build_slate(runtime(), f, o, settings)
    slate["sources"] = status
    return slate, label, demo


def level_name(code) -> str:
    return level_label(code)
