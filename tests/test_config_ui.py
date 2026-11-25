"""Control panel (Config UI).

Tests the logic (validation, diff, surgical application) and the full HTTP
flow on an ephemeral local server: form -> preview -> save.
"""
from __future__ import annotations

import http.client
import shutil
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml

from src.config.schema import Field, all_fields, validate
from src.config.webui import ConfigPanel, create_server


# ---------------------------------------------------------------------- #
# validation                                                              #
# ---------------------------------------------------------------------- #
def test_validate_types_and_limits():
    f = Field("x", "Number", "", kind="float", min_value=-10, max_value=10)
    assert validate(f, "-3,5") == (-3.5, None)        # decimal comma accepted
    assert validate(f, "99")[1] is not None           # above the maximum
    assert validate(f, "hello")[1] is not None        # not numeric
    i = Field("x", "Integer", "", kind="int", min_value=1, max_value=4)
    assert validate(i, "3") == (3, None)
    assert validate(i, "0")[1] is not None
    s = Field("x", "Choice", "", kind="choice", choices=["a", "b"])
    assert validate(s, "a") == ("a", None)
    assert validate(s, "z")[1] is not None
    b = Field("x", "Flag", "", kind="bool")
    assert validate(b, "true") == (True, None)
    assert validate(b, "false") == (False, None)
    v = Field("x", "Empty ok", "", kind="int", empty_ok=True)
    assert validate(v, "") == ("", None)


def test_schema_keys_exist_in_the_real_params():
    """Every schema field must point to a real key of the project's
    params.yaml: if a key gets renamed, this test catches it."""
    pan = ConfigPanel("params.yaml")
    present = {c.key for c in pan.present_fields()}
    expected = {c.key for c in all_fields()}
    assert present == expected, f"missing keys: {expected - present}"


# ---------------------------------------------------------------------- #
# panel logic                                                             #
# ---------------------------------------------------------------------- #
@pytest.fixture
def pan(tmp_path):
    dest = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dest)
    return ConfigPanel(str(dest))


def _base_form(pan: ConfigPanel) -> dict:
    """The form as the browser would send it: all the current values."""
    form = {}
    for c in pan.present_fields():
        v = pan.current_value(c.key)
        if c.kind == "bool":
            form[c.key] = ["false", "true"] if v else ["false"]
        else:
            form[c.key] = ["" if v is None else str(v)]
    return form


def test_no_change_if_form_is_equal(pan):
    changes, errors = pan.read_form(_base_form(pan))
    assert errors == [] and changes == {}


def test_diff_detects_only_the_changed_fields(pan):
    form = _base_form(pan)
    # value DIFFERENT from the current one, whatever it is: the diff must
    # detect the change (the test must not depend on the real params)
    current = pan.current_value("vix.voting_method")
    new = "majority" if current == "gate" else "gate"
    form["vix.voting_method"] = [new]
    form["portfolio.consensus_threshold"] = ["0.7"]
    changes, errors = pan.read_form(form)
    assert errors == []
    assert changes == {"vix.voting_method": new,
                       "portfolio.consensus_threshold": 0.7}


def test_invalid_value_blocks_everything(pan):
    form = _base_form(pan)
    form["portfolio.consensus_threshold"] = ["7"]     # above the maximum (1)
    changes, errors = pan.read_form(form)
    assert len(errors) == 1 and "threshold" in errors[0].lower()


def test_apply_writes_surgically(pan):
    before = Path(pan.editor.path).read_text(encoding="utf-8")
    pan.apply({"vix.voting_method": "gate",
               "notifications.email_enabled": True,
               "vix.tdex_per_quad.2.sell_spx": -4.2})
    text = Path(pan.editor.path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert parsed["vix"]["voting_method"] == "gate"
    assert parsed["notifications"]["email_enabled"] is True
    assert parsed["vix"]["tdex_per_quad"][2]["sell_spx"] == -4.2
    # the other values of quadrant 2 stay
    assert parsed["vix"]["tdex_per_quad"][2]["longvol"] == -3.0
    # the file's comments stay (sample)
    assert "# TDEX staleness guard" in text
    assert "MIDTERM workbook" in text
    # backup of the original
    assert Path(str(pan.editor.path) + ".bak").read_text(
        encoding="utf-8") == before


# ---------------------------------------------------------------------- #
# full HTTP flow                                                          #
# ---------------------------------------------------------------------- #
@pytest.fixture
def web_server(tmp_path):
    dest = tmp_path / "params.yaml"
    shutil.copyfile("params.yaml", dest)
    server, port = create_server(str(dest), port=18766)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    yield port, dest
    server.shutdown()
    server.server_close()


def _req(port: int, method: str, path: str, form: dict | None = None) -> str:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body, headers = None, {}
    if form is not None:
        body = urlencode(form, doseq=True)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    conn.request(method, path, body, headers)
    resp = conn.getresponse()
    out = resp.read().decode("utf-8")
    conn.close()
    return out


def test_flow_form_preview_save(web_server):
    port, dest = web_server
    pan = ConfigPanel(str(dest))

    page = _req(port, "GET", "/")
    assert "Control panel" in page
    assert "majority" in page                         # current value in the form

    form = {k: v[-1] for k, v in _base_form(pan).items()}
    form["vix.tdex_vote_sell_spx"] = "-4.5"
    preview = _req(port, "POST", "/preview", form)
    assert "Preview" in preview and "-4.5" in preview
    assert "vix.tdex_vote_sell_spx" in preview
    # the preview does NOT touch the file
    assert ConfigPanel(str(dest)).current_value("vix.tdex_vote_sell_spx") == -4.0

    saved = _req(port, "POST", "/save", form)
    assert "Saved" in saved
    assert ConfigPanel(str(dest)).current_value("vix.tdex_vote_sell_spx") == -4.5


def test_flow_wrong_value_does_not_save(web_server):
    port, dest = web_server
    pan = ConfigPanel(str(dest))
    before = dest.read_text(encoding="utf-8")
    form = {k: v[-1] for k, v in _base_form(pan).items()}
    form["regime_source.manual_quad"] = "9"           # outside 1..4
    page = _req(port, "POST", "/save", form)
    assert "Fix and try again" in page
    assert dest.read_text(encoding="utf-8") == before  # file untouched
