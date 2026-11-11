"""Keep prices, predictions and the bet log current: one run every PRICE_INTERVAL_MINUTES.

    python scripts/live_loop.py --minutes 50 --git     # what GitHub Actions runs
    python scripts/live_loop.py --once                 # a single refresh, no git

Each tick:
1. pulls the latest commit (so a model retrained by the daily job is picked up);
2. settles finished bets when the daily job has published new results;
3. refreshes fixtures and prices that are due (matches about to start every 10 min,
   later ones less often), rebuilds every prediction, logs new bets and tracks how
   the price of each logged bet has moved since;
4. sends a phone alert for each new bet (optional, see src/atp_edge/notify.py);
5. commits data/predictions when something changed.

At the end it writes continue=true|false to $GITHUB_OUTPUT: the workflow restarts
itself while there are matches coming up, so it doesn't depend on GitHub's cron
(which in practice ran the old hourly job only 3-4 times a day).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atp_edge import tracking  # noqa: E402
from atp_edge.config import PREDICTIONS  # noqa: E402
from atp_edge.data import MATCHES_PATH, load_matches  # noqa: E402
from atp_edge.live import collect  # noqa: E402
from atp_edge.modeling import ENGINE_PATH, MATCH_MODEL  # noqa: E402
from atp_edge.notify import send_new_bets  # noqa: E402
from atp_edge.predict import LATEST_PATH, Runtime, Settings, build_slate  # noqa: E402

STATE = PREDICTIONS / "loop_state.json"


def git(*args, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def file_sig(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else ""


def slate_fingerprint(slate: dict) -> str:
    """What a person would notice: prices, probabilities and bets (not timestamps)."""
    keep = []
    for m in slate.get("matches", []):
        sets = m.get("sets") or {}
        keep.append([m["player_a"], m["player_b"], m.get("start_ts"), round(m["p_model_a"], 4),
                     sorted((m.get("prices_a") or {}).items()), sorted((m.get("prices_b") or {}).items()),
                     (m.get("recommendation") or {}).get("selection"), (m.get("recommendation") or {}).get("price"),
                     sets.get("best_over"), sets.get("best_under"), (sets.get("recommendation") or {}).get("selection")])
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def slim(slate: dict, fixtures: list, odds_events: list) -> dict:
    """Keep latest.json small: it is committed every few minutes."""
    now = time.time()
    horizon = now + 50 * 3600
    keep = lambda e: not e.get("start_ts") or now - 6 * 3600 < float(e["start_ts"]) < horizon  # noqa: E731
    slate["inputs"] = {"fixtures": [f for f in fixtures if keep(f)], "odds_events": [o for o in odds_events if keep(o)]}
    sk = slate.get("skipped", [])
    slate["skipped"] = [{k: s.get(k) for k in ("player_a", "player_b", "tournament", "start_ts", "reason")} for s in sk][:60]
    return slate


def tick(rt: Runtime, state: dict) -> dict:
    out: dict = {}
    if MATCHES_PATH.exists():
        sig = file_sig(MATCHES_PATH)
        if sig != state.get("settled_sig"):
            out["settlement"] = tracking.settle(load_matches())
            state["settled_sig"] = sig
    fixtures, odds_events, status = collect(rt)
    slate = build_slate(rt, fixtures, odds_events, Settings())
    try:
        from atp_edge.narrative import ai_previews
        status["ai_previews"] = ai_previews(slate["matches"])
    except Exception as exc:
        status["ai_previews"] = {"error": str(exc)[:200]}
    status["refreshed_ts"] = time.time()
    slate["sources"] = status
    slate["demo"] = False
    slate = slim(slate, fixtures, odds_events)
    fp = slate_fingerprint(slate)
    changed = fp != state.get("fingerprint")
    stale = time.time() - float(state.get("written_ts") or 0) > 3600
    logged, new_bets = {"skipped": "no price or probability changed"}, []
    if changed or stale:
        # Only when something moved: keeps the git history (one commit per change) small.
        logged = tracking.record(slate)
        new_bets = logged.pop("new_bet_rows", [])
        LATEST_PATH.write_text(json.dumps(slate, separators=(",", ":"), default=str))
        state["fingerprint"], state["written_ts"] = fp, time.time()
    out.update({"matches": len(slate["matches"]), "prices_changed": changed, "logged": logged,
                "alerts": send_new_bets(new_bets), "upcoming_24h": sum(1 for m in slate["matches"]
                                                                      if (m.get("start_ts") or 0) < time.time() + 86400)})
    out["sources"] = {k: v for k, v in status.items() if k != "ai_previews"}
    return out


def commit_and_push(message: str, final: bool) -> str:
    git("add", "-A", "data/predictions")
    if not final:
        # The price cache changes every tick; it only needs saving for the next run.
        git("reset", "-q", "--", "data/predictions/price_cache.json", "data/predictions/loop_state.json")
    if git("diff", "--cached", "--quiet").returncode == 0:
        return "nothing to commit"
    git("commit", "-m", message, check=True)
    for attempt in range(4):
        git("pull", "--rebase", "--autostash", "origin", os.getenv("GITHUB_REF_NAME", "main"))
        if git("push").returncode == 0:
            return "pushed"
        time.sleep(5 * (attempt + 1))
    return "push failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=50, help="how long to keep refreshing")
    ap.add_argument("--interval", type=float, default=float(os.getenv("PRICE_INTERVAL_MINUTES") or 10))
    ap.add_argument("--git", action="store_true", help="pull before and commit/push after each tick")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    interval = max(5.0, args.interval) * 60
    end = time.time() + args.minutes * 60
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        state = {}
    rt, rt_sig, upcoming, ticks = None, None, 0, 0
    while True:
        t0 = time.time()
        final = args.once or t0 + interval > end
        if args.git:
            git("pull", "--rebase", "--autostash", "origin", os.getenv("GITHUB_REF_NAME", "main"))
        sig = (file_sig(ENGINE_PATH), file_sig(MATCH_MODEL))
        if rt is None or sig != rt_sig:
            rt, rt_sig = Runtime(), sig       # a retrained model was committed by the daily job
        try:
            res = tick(rt, state)
            upcoming = res.get("upcoming_24h", 0)
            final = final or upcoming == 0     # nothing to price soon: stop early
            STATE.write_text(json.dumps(state))
            if args.git:
                res["git"] = commit_and_push(f"Live prices {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC", final)
            print(json.dumps(res, default=str), flush=True)
        except Exception:
            traceback.print_exc()
        ticks += 1
        if final:
            break
        time.sleep(max(0.0, interval - (time.time() - t0)))
    keep_going = upcoming > 0
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as fh:
            fh.write(f"continue={'true' if keep_going else 'false'}\n")
    print(f"done: {ticks} refreshes; matches in the next 24 h: {upcoming}; restart: {keep_going}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
