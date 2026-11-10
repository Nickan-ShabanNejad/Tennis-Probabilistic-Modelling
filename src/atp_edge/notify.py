"""Optional phone alerts for new bets (ntfy.sh and/or a Discord webhook).

* NTFY_TOPIC: install the free ntfy app, subscribe to a long random topic name
  (e.g. atp-edge-7f3k29xq), and add the same name as a repo secret.
* DISCORD_WEBHOOK_URL: a channel webhook URL, as a repo secret.

Both are optional; with neither set nothing is sent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from .config import APP_TIMEZONE, secret


def bet_text(b: dict) -> tuple[str, str]:
    when = ""
    try:
        when = datetime.fromtimestamp(float(b["start_ts"]), tz=timezone.utc).astimezone(ZoneInfo(APP_TIMEZONE)).strftime("%a %H:%M")
    except Exception:
        pass
    title = f"BET {b['selection']} @ {float(b['price']):.2f} ({b.get('book') or '?'})"
    body = (f"{b['player_a']} vs {b['player_b']} · {b.get('tournament') or ''} {when}\n"
            f"{b['market']} · model {float(b['probability']):.0%} · EV {float(b['ev']):+.1%} · "
            f"stake {float(b['stake']):.2%} of bankroll")
    return title, body


def send_new_bets(bets: list[dict]) -> dict:
    topic, hook = secret("NTFY_TOPIC"), secret("DISCORD_WEBHOOK_URL")
    if not bets or not (topic or hook):
        return {"sent": 0, "enabled": bool(topic or hook)}
    sent, errors = 0, []
    for b in bets[:10]:
        title, body = bet_text(b)
        try:
            if topic:
                requests.post(f"https://ntfy.sh/{topic.strip()}", data=body.encode(),
                              headers={"Title": title.encode("ascii", "ignore").decode(), "Tags": "tennis"}, timeout=15)
            if hook:
                requests.post(hook.strip(), json={"content": f"**{title}**\n{body}"}, timeout=15)
            sent += 1
        except Exception as exc:
            errors.append(str(exc)[:120])
    return {"sent": sent, "enabled": True, "errors": errors[:2]}
