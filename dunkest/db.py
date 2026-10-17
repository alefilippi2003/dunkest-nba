"""Accesso al database SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd


def connect(path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def table_exists(con, name: str) -> bool:
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (name,)).fetchone() is not None


def save_season(con, table: str, df: pd.DataFrame, season: str) -> None:
    """Sostituisce i dati di una stagione in una tabella (idempotente: rilanciarlo non duplica)."""
    if df is None or df.empty:
        print(f"    (nessun dato per {table} {season})")
        return
    df = df.copy()
    df["SEASON"] = season
    if table_exists(con, table):
        con.execute(f'DELETE FROM "{table}" WHERE SEASON = ?', (season,))
    df.to_sql(table, con, if_exists="append", index=False)
    con.commit()
    print(f"    {table}: {len(df)} righe")
