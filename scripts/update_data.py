"""Daily data refresh: results, rankings, court speed, closing odds.

Runs in GitHub Actions (see .github/workflows/daily.yml). Every source is
best-effort: a failure is reported in data/generated/data_status.json and the
run continues with what it has.
"""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from atp_edge import court_speed, odds_history  # noqa: E402
from atp_edge.config import GENERATED, secret  # noqa: E402
from atp_edge.data import build_database, fetch_tml_years, load_matches  # noqa: E402
from atp_edge.rankings import RANKINGS_PATH  # noqa: E402


def main() -> int:
    status = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    year = date.today().year
    extra = []

    tml, tml_status = fetch_tml_years([year - 1, year])
    status["tennismylife"] = tml_status
    if not tml.empty:
        extra.append(tml)

    if secret("RAPIDAPI_KEY"):
        try:
            from atp_edge.rapidapi import MatchstatClient, map_ids_to_archive
            client = MatchstatClient()
            ranks = client.rankings(500)
            if not ranks.empty:
                ranks.to_csv(RANKINGS_PATH, index=False)
            recent = client.recent_results(days=14)
            archive = load_matches()
            recent = map_ids_to_archive(recent, archive)
            extra.append(recent)
            status["matchstat"] = {"ok": True, "rankings": int(len(ranks)), "recent_results": int(len(recent)), "api_calls": client.calls}
        except Exception as exc:
            status["matchstat"] = {"ok": False, "error": str(exc)}
    else:
        status["matchstat"] = {"ok": False, "error": "RAPIDAPI_KEY not set"}

    # Court speed: current 52 weeks every run (one polite request).
    try:
        fresh = court_speed.fetch_tennis_abstract([], pause=0)
        table = court_speed.merge_into_table(fresh)
        status["court_speed"] = {"ok": not fresh.empty, "rows": int(len(table)), "fetched": int(len(fresh))}
    except Exception as exc:
        status["court_speed"] = {"ok": False, "error": str(exc)}

    status["closing_odds"] = odds_history.refresh_current_season()

    matches, report = build_database(extra)
    status["database"] = report
    # Bets are settled by the live-prices job as soon as this commit lands.
    status["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (GENERATED / "data_status.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
