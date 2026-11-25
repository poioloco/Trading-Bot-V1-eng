"""Saving and loading the engines' state (indicator_state table).

The state is what makes the incremental update possible: instead of
recomputing from 2005 on every run, the engines restart from here. It
contains the recursive seeds (EMA-style averages/RSI) plus a small window
of recent days, just enough to fill the rolling windows and to guarantee
numbers identical to a full recompute.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from ..data.database import Database


def save_state(db: Database, engine: str, as_of: date, state: dict) -> None:
    payload = json.dumps(state)
    db.con.execute(
        "INSERT INTO indicator_state (engine, as_of, state_json) VALUES (?, ?, ?) "
        "ON CONFLICT (engine, as_of) DO UPDATE SET state_json = excluded.state_json",
        [engine, as_of, payload],
    )
    # keep only the latest state per engine (history not needed)
    db.con.execute(
        "DELETE FROM indicator_state WHERE engine = ? AND as_of < ?", [engine, as_of]
    )


def load_state(db: Database, engine: str) -> tuple[date, dict] | None:
    row = db.con.execute(
        "SELECT as_of, state_json FROM indicator_state WHERE engine = ? "
        "ORDER BY as_of DESC LIMIT 1",
        [engine],
    ).fetchone()
    if not row:
        return None
    return row[0], json.loads(row[1])


# ---- helpers to serialise a window of rows (DataFrame) ------------------ #
def frame_to_json(df: pd.DataFrame) -> dict:
    return {
        "index": [pd.Timestamp(i).isoformat() for i in df.index],
        "columns": list(df.columns),
        "data": df.to_numpy().tolist(),
    }


def frame_from_json(obj: dict) -> pd.DataFrame:
    df = pd.DataFrame(obj["data"], columns=obj["columns"])
    df.index = pd.to_datetime(obj["index"])
    return df
