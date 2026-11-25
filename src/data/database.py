"""Connection to the local DuckDB database and write operations.

The whole history lives in a single `.duckdb` file. Writes are
**idempotent**: re-running an update over the same days does not create
duplicate rows (symbol+date key). Raw prices are never edited by hand:
they are only overwritten by re-downloading the same range.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = "data/market.duckdb"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Database:
    """Thin wrapper around a DuckDB connection with a guaranteed schema."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        self.con.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Price writes (idempotent per range)                                 #
    # ------------------------------------------------------------------ #
    def upsert_prices(
        self, df: pd.DataFrame, symbol: str, source: str, synthetic: bool = False
    ) -> int:
        """`df`: date index, open/high/low/close/volume columns. Replaces
        the existing rows of `symbol` in the same range (no duplicates)."""
        if df is None or df.empty:
            return 0
        rows = df.copy()
        rows.index = pd.to_datetime(rows.index).normalize()
        rows = rows[~rows.index.duplicated(keep="last")].sort_index()
        payload = pd.DataFrame(
            {
                "symbol": symbol,
                "date": rows.index.date,
                "open": rows["open"].astype(float).to_numpy(),
                "high": rows["high"].astype(float).to_numpy(),
                "low": rows["low"].astype(float).to_numpy(),
                "close": rows["close"].astype(float).to_numpy(),
                "volume": rows.get("volume", pd.Series(0.0, index=rows.index))
                .astype(float)
                .to_numpy(),
                "source": source,
                "fetched_at": _now(),
                "synthetic": bool(synthetic),
            }
        )
        dmin, dmax = payload["date"].min(), payload["date"].max()
        self.con.register("payload_view", payload)
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute(
                "DELETE FROM prices_raw WHERE symbol = ? AND date BETWEEN ? AND ?",
                [symbol, dmin, dmax],
            )
            self.con.execute("INSERT INTO prices_raw SELECT * FROM payload_view")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("payload_view")
        return len(payload)

    def upsert_corporate_actions(
        self, actions: pd.DataFrame, source: str
    ) -> int:
        """`actions`: columns symbol, date, action_type, value."""
        if actions is None or actions.empty:
            return 0
        payload = actions.copy()
        payload["date"] = pd.to_datetime(payload["date"]).dt.date
        payload["source"] = source
        payload["fetched_at"] = _now()
        payload = payload[
            ["symbol", "date", "action_type", "value", "source", "fetched_at"]
        ]
        self.con.register("ca_view", payload)
        self.con.execute(
            "INSERT INTO corporate_actions SELECT * FROM ca_view "
            "ON CONFLICT (symbol, date, action_type) DO UPDATE SET "
            "value = excluded.value, source = excluded.source, "
            "fetched_at = excluded.fetched_at"
        )
        self.con.unregister("ca_view")
        return len(payload)

    # ------------------------------------------------------------------ #
    # Run and signal logs (heartbeat + idempotency)                       #
    # ------------------------------------------------------------------ #
    def start_run(self, kind: str, as_of: date | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.con.execute(
            "INSERT INTO runs_log (run_id, kind, status, as_of, started_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            [run_id, kind, as_of, _now()],
        )
        return run_id

    def finish_run(
        self, run_id: str, status: str, message: str | None = None
    ) -> None:
        self.con.execute(
            "UPDATE runs_log SET status = ?, finished_at = ?, message = ? "
            "WHERE run_id = ?",
            [status, _now(), message, run_id],
        )

    def last_successful_run(self, kind: str | None = None) -> dict | None:
        q = "SELECT * FROM runs_log WHERE status = 'ok'"
        params: list = []
        if kind:
            q += " AND kind = ?"
            params.append(kind)
        q += " ORDER BY started_at DESC LIMIT 1"
        row = self.con.execute(q, params).fetchdf()
        return None if row.empty else row.iloc[0].to_dict()

    def write_signal(
        self, as_of: date, asset: str, action: str, quad: int | None
    ) -> bool:
        """Idempotent: returns False if today's signal for that asset already
        exists (so duplicate notifications are not re-sent)."""
        existing = self.con.execute(
            "SELECT 1 FROM signals_log WHERE as_of = ? AND asset = ?",
            [as_of, asset],
        ).fetchone()
        if existing:
            return False
        self.con.execute(
            "INSERT INTO signals_log (as_of, asset, action, quad, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [as_of, asset, action, quad, _now()],
        )
        return True

    # ------------------------------------------------------------------ #
    # Service reads                                                       #
    # ------------------------------------------------------------------ #
    def symbols(self) -> list[str]:
        df = self.con.execute(
            "SELECT DISTINCT symbol FROM prices_raw ORDER BY symbol"
        ).fetchdf()
        return df["symbol"].tolist()

    def coverage(self) -> pd.DataFrame:
        """For each symbol: first/last date, number of bars, whether synthetic."""
        return self.con.execute(
            "SELECT symbol, MIN(date) AS first, MAX(date) AS last, "
            "COUNT(*) AS bars, BOOL_OR(synthetic) AS synthetic "
            "FROM prices_raw GROUP BY symbol ORDER BY symbol"
        ).fetchdf()
