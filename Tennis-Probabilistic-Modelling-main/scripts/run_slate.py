"""One refresh of fixtures, prices, predictions and the bet log.

    python scripts/run_slate.py            # live (needs RAPIDAPI_KEY and/or ODDS_API_KEY); same as live_loop.py --once
    python scripts/run_slate.py --demo     # offline demo with synthetic prices (nothing is logged)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atp_edge.live import collect  # noqa: E402
from atp_edge.predict import LATEST_PATH, Runtime, Settings, build_slate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if not args.demo:
        import live_loop
        sys.argv = [sys.argv[0], "--once"]
        return live_loop.main()
    rt = Runtime()
    fixtures, odds_events, status = collect(rt, demo=True)
    slate = build_slate(rt, fixtures, odds_events, Settings())
    slate["sources"] = status
    slate["inputs"] = {"fixtures": fixtures, "odds_events": odds_events}
    slate["demo"] = True
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(slate, separators=(",", ":"), default=str))
    print(json.dumps({"matches": len(slate["matches"]), "skipped": len(slate["skipped"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
