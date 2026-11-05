"""Current ATP rankings (refreshed daily from Matchstat) applied on top of ratings."""
from __future__ import annotations

import pandas as pd

from .config import GENERATED

RANKINGS_PATH = GENERATED / "rankings.csv"


def apply_current_rankings(engine) -> int:
    if not RANKINGS_PATH.exists():
        return 0
    df = pd.read_csv(RANKINGS_PATH)
    return engine.apply_rankings(zip(df["player"], df["rank"], df.get("points", pd.Series(0, index=df.index))))
