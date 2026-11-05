"""Rebuild ratings from the full match history and retrain every model.

    python scripts/train.py            # ~2-4 minutes
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atp_edge.court_speed import CourtSpeed  # noqa: E402
from atp_edge.data import build_database  # noqa: E402
from atp_edge.engine import Engine  # noqa: E402
from atp_edge.history import replay  # noqa: E402
from atp_edge.modeling import train_all  # noqa: E402
from atp_edge.rankings import apply_current_rankings  # noqa: E402


def main():
    matches, report = build_database()
    print("data:", report, flush=True)
    engine = Engine(CourtSpeed())
    rows = replay(matches, engine)
    report["current_rankings_applied"] = apply_current_rankings(engine)
    metrics = train_all(rows, engine, report)
    summary = {k: metrics[k] for k in ("selected_model", "latest_match_date")}
    summary["benchmark"] = {k: metrics["market_benchmark"][k] for k in ("model_log_loss", "pinnacle_log_loss", "matches")}
    summary["blend"] = metrics["market_blend"]
    summary["sets"] = {k: metrics["sets_model"].get(k) for k in ("log_loss", "base_rate_log_loss", "eval_matches")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
