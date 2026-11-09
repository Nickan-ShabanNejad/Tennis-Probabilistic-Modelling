"""Fetch upcoming fixtures and prices, predict, recommend, and log.

    python scripts/run_slate.py            # live (needs RAPIDAPI_KEY and/or ODDS_API_KEY)
    python scripts/run_slate.py --demo     # offline demo with synthetic prices
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atp_edge import tracking  # noqa: E402
from atp_edge.config import secret  # noqa: E402
from atp_edge.live import collect  # noqa: E402
from atp_edge.predict import LATEST_PATH, Runtime, Settings, build_slate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    rt = Runtime()
    fixtures, odds_events, status = collect(rt, demo=args.demo)
    slate = build_slate(rt, fixtures, odds_events, Settings())
    if not args.demo:
        from atp_edge.narrative import ai_previews
        status["ai_previews"] = ai_previews(slate["matches"])
    slate["sources"] = status
    slate["inputs"] = {"fixtures": fixtures, "odds_events": odds_events}
    slate["demo"] = bool(args.demo)
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(slate, indent=1, default=str))
    logged = {} if (args.demo or args.no_log) else tracking.record(slate)
    print(json.dumps({"matches": len(slate["matches"]), "skipped": len(slate["skipped"]),
                      "bets": sum(1 for m in slate["matches"] if m.get("recommendation") or (m.get("sets") or {}).get("recommendation")),
                      "sources": status, "logged": logged}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
