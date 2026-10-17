"""Percorsi e costanti del progetto."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

DB_PATH = DATA_DIR / "dunkest.db"
MODEL_PATH = MODELS_DIR / "dunkest_models.pkl"

FIRST_SEASON = "2022-23"          # prima stagione scaricata con `update --all`
SEASON_TYPE = "Regular Season"


def season_of(date) -> str:
    """Stagione NBA di una data: 2026-10-21 -> '2026-27' (da agosto in poi conta la nuova)."""
    d = pd.Timestamp(date)
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def seasons_between(first: str, last: str) -> list[str]:
    """['2022-23', '2023-24', ...] da first a last inclusi."""
    y0, y1 = int(first[:4]), int(last[:4])
    return [f"{y}-{str(y + 1)[2:]}" for y in range(y0, y1 + 1)]
