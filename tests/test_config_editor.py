"""Surgical editor of params.yaml and the history_end boundary.

The editor must change ONLY the line of the requested key: comments,
formatting and all the rest of the file stay identical.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.config import ParamsEditor

YAML_DEMO = """\
# header comment
data:
  # comment of the mode
  mode: live   # inline comment
  history_end: "2025-12-30"

vix:
  voting_method: majority
  voting_legs:
    vrp: true
    rsi: true
    tdex: true
  tdex_max_staleness_days: 10

legs:
  market: true
  tdex: true    # note
"""


@pytest.fixture
def yml(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text(YAML_DEMO, encoding="utf-8")
    return p


def test_set_scalar_keeps_comments(yml):
    ed = ParamsEditor(yml)
    ed.set("data.mode", "demo")
    text = yml.read_text(encoding="utf-8")
    assert "mode: demo   # inline comment" in text
    assert "# comment of the mode" in text
    assert "# header comment" in text
    assert yaml.safe_load(text)["data"]["mode"] == "demo"
    assert Path(str(yml) + ".bak").exists()           # backup created


def test_set_nested_key(yml):
    ed = ParamsEditor(yml)
    ed.set("vix.voting_legs.tdex", False)
    parsed = yaml.safe_load(yml.read_text(encoding="utf-8"))
    assert parsed["vix"]["voting_legs"]["tdex"] is False
    assert parsed["legs"]["tdex"] is True             # the namesake is NOT touched


def test_set_date_and_empty(yml):
    ed = ParamsEditor(yml)
    ed.set("data.history_end", "2026-07-03")
    assert str(ed.value("data.history_end")) == "2026-07-03"
    ed.set("vix.tdex_max_staleness_days", "")
    parsed = yaml.safe_load(yml.read_text(encoding="utf-8"))
    assert parsed["vix"]["tdex_max_staleness_days"] == ""


def test_missing_key_does_not_touch_the_file(yml):
    ed = ParamsEditor(yml)
    before = yml.read_text(encoding="utf-8")
    with pytest.raises(KeyError):
        ed.set("vix.does_not_exist", 1)
    assert yml.read_text(encoding="utf-8") == before


def test_editor_on_the_real_params(tmp_path):
    """The project's REAL params.yaml: edit one key and check that the rest
    of the file (comments included) stays byte-for-byte identical."""
    dest = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dest)
    before = dest.read_text(encoding="utf-8").splitlines()
    # value DIFFERENT from the current one, otherwise the editor (rightly)
    # touches nothing and the test measures nothing
    current = yaml.safe_load(dest.read_text(encoding="utf-8"))["vix"]["voting_method"]
    new = "majority" if current == "gate" else "gate"
    ed = ParamsEditor(dest)
    ed.set("vix.voting_method", new)
    after = dest.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(diff) == 1                             # ONE single line changed
    assert f"voting_method: {new}" in after[diff[0]]
    parsed = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert parsed["vix"]["voting_method"] == new


# ---------------------------------------------------------------------- #
# history_end moved forward by update-data                                #
# ---------------------------------------------------------------------- #
def test_update_history_end(tmp_path):
    import pandas as pd
    from cli import _update_history_end
    from src.data.database import Database

    dest = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dest)
    db = Database(tmp_path / "m.duckdb")
    # dates ALWAYS beyond the current boundary: update-data moves
    # history_end forward in the real params.yaml, so fixed dates would age
    cutoff = yaml.safe_load(dest.read_text(encoding="utf-8"))["data"].get(
        "history_end") or "2026-06-01"
    idx = pd.bdate_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=5)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 0.0}, index=idx)
    db.upsert_prices(df, "^SPX", "test")              # REAL data
    new = _update_history_end(str(dest), db)
    assert new == idx[-1].date().isoformat()
    ed = ParamsEditor(dest)
    assert str(ed.value("data.history_end")) == new
    # second call: already up to date, no change
    assert _update_history_end(str(dest), db) is None
    db.close()


def test_history_end_ignores_synthetic_data(tmp_path):
    import pandas as pd
    from cli import _update_history_end
    from src.data.database import Database

    dest = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dest)
    db = Database(tmp_path / "m.duckdb")
    idx = pd.bdate_range("2027-01-01", periods=3)
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 0.0}, index=idx)
    db.upsert_prices(df, "^SPX", "demo", synthetic=True)
    # only synthetic data in the future: the boundary does NOT move
    assert _update_history_end(str(dest), db) is None
    db.close()
