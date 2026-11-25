"""The incremental state must notice RETROACTIVE changes to the history.

Reproduces the defect seen in production: warm-up with the history stopped
at a certain date, then a backfill of past months; the old step() glued the
new days onto smoothers that had fallen behind, giving a regime different
from the full recompute (the one of the backtest/dashboard).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.app import (build_database, build_pipeline, build_store,
                     etf_symbols, load_params)
from src.data.ingest import seed_offline_demo


@pytest.fixture()
def env(tmp_path):
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp_path / "m.duckdb")
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2019, 1, 2), date(2021, 6, 30))
    yield p, db, build_store(db)
    db.close()


def test_rebuild_after_backfill(env):
    p, db, store = env
    pipe = build_pipeline(p, db, store)

    rep1 = pipe.run(date(2021, 6, 30))
    assert rep1.mode == "warm-up"

    # same day, nothing new: uses the saved state
    pipe2 = build_pipeline(p, db, store)
    rep2 = pipe2.run(date(2021, 6, 30))
    assert rep2.mode == "up to date"
    assert rep2.quad == rep1.quad

    # retroactive BACKFILL: correct a close one year earlier
    db.con.execute(
        "UPDATE prices_raw SET close = close * 1.5 "
        "WHERE symbol = '^SPX' AND date = '2020-06-15'")

    pipe3 = build_pipeline(p, db, store)
    rep3 = pipe3.run(date(2021, 6, 30))
    assert rep3.mode == "rebuild"                # state thrown away and redone

    # and the result matches the forced full recompute
    pipe4 = build_pipeline(p, db, store)
    rep4 = pipe4.run(date(2021, 6, 30), force_rebuild=True)
    assert rep3.quad == rep4.quad
    assert rep3.asset_signals == rep4.asset_signals
    assert rep3.metrics == pytest.approx(rep4.metrics)


def test_rebuild_old_state_without_fingerprint(env):
    """States saved BEFORE this change have no fingerprint: they must be
    rebuilt once, then it goes back to incremental."""
    p, db, store = env
    pipe = build_pipeline(p, db, store)
    pipe.run(date(2021, 6, 30))

    # simulate an "old" state: remove the fingerprint from the saved JSON
    import json
    for eng in ("regime", "vix"):
        as_of, st = db.con.execute(
            "SELECT as_of, state_json FROM indicator_state WHERE engine = ?",
            [eng]).fetchone()
        d = json.loads(st)
        d.pop("_data_fingerprint", None)
        db.con.execute(
            "UPDATE indicator_state SET state_json = ? WHERE engine = ?",
            [json.dumps(d), eng])

    rep = build_pipeline(p, db, store).run(date(2021, 6, 30))
    assert rep.mode == "rebuild"

    # the next run has the fingerprint again: no further rebuild
    rep2 = build_pipeline(p, db, store).run(date(2021, 6, 30))
    assert rep2.mode == "up to date"


def test_metrics_approx_helper():
    """pytest.approx on a dict with None is not obvious: check of the
    comparison used above (spx/vix float, tdex may be None in both)."""
    a = {"spx": 1.0, "tdex_smooth": None}
    assert a == pytest.approx({"spx": 1.0, "tdex_smooth": None})
