"""Paths and secrets. Secrets are read from environment variables or Streamlit
secrets - never hard-code keys in this repository."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
SEED = DATA / "seed"
REFERENCE = DATA / "reference"
ODDS = DATA / "odds"
GENERATED = DATA / "generated"
PREDICTIONS = DATA / "predictions"
MODEL_DIR = ROOT / "model"

for _p in (RAW, GENERATED, PREDICTIONS, MODEL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

SECRET_NAMES = ("RAPIDAPI_KEY", "ODDS_API_KEY")


def secret(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets (if running in Streamlit)."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    # Backwards-compatible name used by the previous repository.
    if name == "RAPIDAPI_KEY" and os.getenv("MATCHSTAT_API_KEY"):
        return os.getenv("MATCHSTAT_API_KEY", "").strip()
    try:
        import streamlit as st  # noqa: WPS433 - optional dependency at runtime

        value = str(st.secrets.get(name, "") or "").strip()
        if not value and name == "RAPIDAPI_KEY":
            value = str(st.secrets.get("MATCHSTAT_API_KEY", "") or "").strip()
        return value or default
    except Exception:
        return default


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Toronto")
