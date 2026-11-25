"""Tests of the HTML dashboard (presentation only, synthetic demo DB)."""
from __future__ import annotations

from datetime import date

import pytest

from src.app import build_database, build_store, etf_symbols, load_params
from src.dashboard import build_dashboard
from src.data.ingest import seed_offline_demo


@pytest.fixture()
def demo(tmp_path):
    p = load_params("params.yaml")
    p["data"]["database"] = str(tmp_path / "demo.duckdb")
    db = build_database(p)
    seed_offline_demo(db, etf_symbols(p), date(2020, 1, 2), date(2021, 6, 30))
    yield p, db, build_store(db), tmp_path
    db.close()


def test_dashboard_is_generated(demo):
    p, db, store, tmp_path = demo
    # hermetic: no deposits, even if contributions.csv exists in the project
    p.setdefault("portfolio", {})["contributions_file"] = str(
        tmp_path / "no_contributions.csv")
    out = build_dashboard(p, db, store, date(2021, 6, 30),
                          out_path=tmp_path / "dashboard.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # consensus signal clearly visible
    assert any(s in html for s in ("RISK-ON", "RISK-OFF", "CASH"))
    # legs' votes and regime
    for s in ("MARKET leg", "VOLATILITY leg", "TDEX leg",
              "Current regime"):
        assert s in html
    # interactive charts (ECharts) and demo data warning
    assert 'id="chbt"' in html and "window.BT=" in html
    assert "DEMO" in html
    # exposure section: how much time in equities, in TLT, in cash
    assert "Where is the capital" in html
    assert "Year by year" in html and "expbar" in html
    # no deposits: the P&L section invites you to record them
    assert "No deposits recorded" in html


def test_dashboard_with_deposits(demo, monkeypatch):
    p, db, store, tmp_path = demo
    contr = tmp_path / "contributions.csv"
    contr.write_text("date,amount_eur\n2021-01-04,10000.00\n",
                     encoding="utf-8")
    p.setdefault("portfolio", {})["contributions_file"] = str(contr)
    out = build_dashboard(p, db, store, date(2021, 6, 30),
                          out_path=tmp_path / "dash2.html")
    html = out.read_text(encoding="utf-8")
    assert "Deposited" in html and "Value today" in html
    assert "10,000" in html          # amount format
    # comparison with the benchmark and full trade log
    assert "S&P 500 (same deposits)" in html
    assert "MFE" in html and "MAE" in html
    # interactive charts for both groups
    assert 'id="chbt"' in html and 'id="chpf"' in html
    assert "window.BT=" in html and "window.PF=" in html


def test_dashboard_slider_and_dates(demo):
    """First-deposit slider present, with the data for the JS simulator;
    dates shown in dd/mm/yyyy format."""
    import json
    import re
    p, db, store, tmp_path = demo
    contr = tmp_path / "contributions.csv"
    contr.write_text("date,amount_eur\n2021-01-04,1000\n", encoding="utf-8")
    p.setdefault("portfolio", {})["contributions_file"] = str(contr)
    out = build_dashboard(p, db, store, date(2021, 6, 30),
                          out_path=tmp_path / "dash3.html")
    html = out.read_text(encoding="utf-8")
    assert 'id="sim-range"' in html and 'id="chcagr"' in html
    blob = json.loads(re.search(r"window\.PF=(\{.*?\});</script>", html).group(1))
    n = len(blob["date"])
    assert n > 100 and len(blob["tgt"]) == n
    assert len(blob["on"]) == n and len(blob["off"]) == n and len(blob["reg"]) == n
    assert blob["plan"] == [["2021-01-04", 1000.0]]
    # date of the first deposit in the slider and in the log: dd/mm/yyyy
    assert ">04/01/2021<" in html
    assert "Market data as of <b>30/06/2021</b>" in html
